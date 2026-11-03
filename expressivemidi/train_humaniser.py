"""
Train the humaniser on the ASAP-derived dataset (see expressivemidi/data/asap.py).

Works on CPU or GPU. Feature/target normalisation statistics are stored *inside* the
checkpoint so inference uses exactly the training scale.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data.asap import N_TARGETS, REGRESSION_TARGETS, TARGET_NAMES
from .features import FEATURE_NAMES
from .models.humaniser_net import HumaniserNet


def load_split(path):
    d = np.load(path, allow_pickle=False)
    return d["X"], d["Y"], d["M"], d["lengths"]


class WindowDataset(Dataset):
    """Fixed-length windows that never cross piece boundaries."""

    def __init__(self, X, Y, M, lengths, x_mean, x_std, y_mean, y_std, seq_len=128, stride=64):
        self.X = ((X - x_mean) / x_std).astype(np.float32)
        Yn = Y.copy()
        Yn[:, :REGRESSION_TARGETS] = (Y[:, :REGRESSION_TARGETS] - y_mean) / y_std
        self.Y, self.M = Yn.astype(np.float32), M.astype(np.float32)
        self.seq_len = seq_len
        self.index = []
        start = 0
        for n in lengths:
            for s in range(0, max(n - seq_len, 0) + 1, stride):
                self.index.append((start + s, min(seq_len, n - s)))
            if n > seq_len and (n - seq_len) % stride:
                self.index.append((start + n - seq_len, seq_len))
            start += n

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, n = self.index[i]
        L = self.seq_len
        x = np.zeros((L, self.X.shape[1]), np.float32)
        y = np.zeros((L, N_TARGETS), np.float32)
        m = np.zeros((L, N_TARGETS), np.float32)
        x[:n], y[:n], m[:n] = self.X[s:s + n], self.Y[s:s + n], self.M[s:s + n]
        pad = np.arange(L) >= n
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), torch.from_numpy(pad)


def masked_loss(pred, y, m):
    r = REGRESSION_TARGETS
    reg = F.huber_loss(pred[..., :r], y[..., :r], reduction="none", delta=1.0) * m[..., :r]
    bce = F.binary_cross_entropy_with_logits(pred[..., r:], y[..., r:], reduction="none") * m[..., r:]
    per_target = torch.cat([reg, bce], -1).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)
    return per_target.sum(), per_target


@torch.no_grad()
def evaluate(model, loader, y_mean, y_std, device):
    """Metrics in real units, plus the same metrics for a 'no expression' baseline."""
    model.eval()
    sums = {}
    def add(k, v, n):
        a, b = sums.get(k, (0.0, 0))
        sums[k] = (a + v, b + n)
    for x, y, m, pad in loader:
        x, y, m, pad = x.to(device), y.to(device), m.to(device), pad.to(device)
        p = model(x, pad)
        for k, name in enumerate(TARGET_NAMES[:REGRESSION_TARGETS]):
            sel = m[..., k] > 0
            pr = p[..., k][sel] * y_std[k] + y_mean[k]
            gt = y[..., k][sel] * y_std[k] + y_mean[k]
            add(f"{name}_mae", (pr - gt).abs().sum().item(), sel.sum().item())
            add(f"{name}_baseline_mae", (gt - y_mean[k]).abs().sum().item(), sel.sum().item())
            add(f"{name}_pg", (pr * gt).sum().item(), 0)
            add(f"{name}_pp", (pr * pr).sum().item(), 0)
            add(f"{name}_gg", (gt * gt).sum().item(), 0)
            add(f"{name}_p", pr.sum().item(), 0)
            add(f"{name}_g", gt.sum().item(), 0)
            add(f"{name}_n", sel.sum().item(), 0)
        for k, name in enumerate(TARGET_NAMES[REGRESSION_TARGETS:], start=REGRESSION_TARGETS):
            sel = m[..., k] > 0
            pr = (p[..., k][sel] > 0).float()
            gt = y[..., k][sel]
            add(f"{name}_acc", (pr == gt).float().sum().item(), sel.sum().item())
            add(f"{name}_baseline_acc", (gt == (gt.mean() > 0.5).float()).float().sum().item(), sel.sum().item())
    out = {}
    for k, (v, n) in sums.items():
        if n:
            out[k] = v / n
    for name in TARGET_NAMES[:REGRESSION_TARGETS]:
        n = sums[f"{name}_n"][0]
        cov = sums[f"{name}_pg"][0] / n - (sums[f"{name}_p"][0] / n) * (sums[f"{name}_g"][0] / n)
        vp = sums[f"{name}_pp"][0] / n - (sums[f"{name}_p"][0] / n) ** 2
        vg = sums[f"{name}_gg"][0] / n - (sums[f"{name}_g"][0] / n) ** 2
        out[f"{name}_corr"] = cov / max(np.sqrt(vp * vg), 1e-9)
    return out


def train(data_dir="data/processed", out_path="models/humaniser.pt", epochs=30, batch_size=64,
          lr=5e-4, patience=6, seq_len=128, d_model=128, num_layers=3, threads=None, max_minutes=None):
    if threads:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    Xtr, Ytr, Mtr, Ltr = load_split(os.path.join(data_dir, "asap_train.npz"))
    Xva, Yva, Mva, Lva = load_split(os.path.join(data_dir, "asap_validation.npz"))

    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-6
    r = REGRESSION_TARGETS
    y_mean = np.array([(Ytr[:, k][Mtr[:, k] > 0]).mean() for k in range(r)])
    y_std = np.array([(Ytr[:, k][Mtr[:, k] > 0]).std() + 1e-6 for k in range(r)])

    tr = WindowDataset(Xtr, Ytr, Mtr, Ltr, x_mean, x_std, y_mean, y_std, seq_len, seq_len // 2)
    va = WindowDataset(Xva, Yva, Mva, Lva, x_mean, x_std, y_mean, y_std, seq_len, seq_len)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    print(f"device={device} train windows={len(tr)} val windows={len(va)}")

    model = HumaniserNet(len(FEATURE_NAMES), N_TARGETS, d_model=d_model, num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps, pct_start=0.1)
    ym_t = torch.tensor(y_mean, device=device, dtype=torch.float32)
    ys_t = torch.tensor(y_std, device=device, dtype=torch.float32)

    best, bad, history = float("inf"), 0, []
    t0 = time.time()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for x, y, m, pad in tl:
            x, y, m, pad = x.to(device), y.to(device), m.to(device), pad.to(device)
            loss, _ = masked_loss(model(x, pad), y, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vloss = sum(masked_loss(model(x.to(device), pad.to(device)), y.to(device), m.to(device))[0].item()
                        for x, y, m, pad in vl) / len(vl)
        metrics = evaluate(model, vl, ym_t, ys_t, device)
        history.append(dict(epoch=epoch + 1, train_loss=tot / len(tl), val_loss=vloss, **metrics))
        print(f"epoch {epoch+1:>2} train={tot/len(tl):.4f} val={vloss:.4f} "
              f"micro MAE={metrics['micro_mae']:.1f}ms (baseline {metrics['micro_baseline_mae']:.1f}) "
              f"vel corr={metrics['vel_corr']:.2f} tempo corr={metrics['tempo_corr']:.2f} "
              f"pedal acc={metrics.get('pedal_acc', 0):.2f} [{(time.time()-t0)/60:.1f} min]", flush=True)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            torch.save(dict(state_dict=model.state_dict(), config=model.config,
                            feature_names=FEATURE_NAMES, target_names=TARGET_NAMES,
                            x_mean=x_mean.tolist(), x_std=x_std.tolist(),
                            y_mean=y_mean.tolist(), y_std=y_std.tolist(), seq_len=seq_len,
                            val_metrics=metrics, epoch=epoch + 1), out_path)
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            print("time budget reached")
            break
    with open(os.path.splitext(out_path)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return out_path


def test(model_path="models/humaniser.pt", data_dir="data/processed"):
    """Report held-out test metrics (pieces never seen in training/validation)."""
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = HumaniserNet(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    X, Y, M, L = load_split(os.path.join(data_dir, "asap_test.npz"))
    ds = WindowDataset(X, Y, M, L, np.array(ck["x_mean"]), np.array(ck["x_std"]),
                       np.array(ck["y_mean"]), np.array(ck["y_std"]), ck["seq_len"], ck["seq_len"])
    m = evaluate(model, DataLoader(ds, batch_size=128), torch.tensor(ck["y_mean"]).float(),
                 torch.tensor(ck["y_std"]).float(), torch.device("cpu"))
    return m
