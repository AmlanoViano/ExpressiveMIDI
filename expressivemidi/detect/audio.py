"""
AI-generated audio detector (Suno, Udio, MusicGen, Stable Audio, ... vs human-made music).

Approach
    Log-mel spectrogram excerpts -> compact residual CNN -> per-excerpt logit; a track's
    score is the mean logit over excerpts. Current generators synthesise audio through
    neural codecs / vocoders that leave spectral fingerprints (band-limited energy,
    periodic upsampling artefacts, over-smooth high frequencies) that a CNN picks up
    reliably *for the generators it has seen*.

Honest limitations (also printed by the CLI)
    * Detectors generalise poorly to generators missing from training. Use
      `--holdout-source` to measure this before trusting the model.
    * Heavy re-processing (low-bitrate MP3, mastering chains, recording off speakers)
      degrades accuracy. Robustness augmentation reduces but does not remove this.
    * A probability is evidence, not proof. Never use it alone to accuse anyone.

Data layout expected by `prepare`:
    data/audio/real/<source>/**/*.mp3|wav|flac|ogg|m4a
    data/audio/ai/<source>/**/*      (e.g. ai/suno, ai/udio, ai/musicgen)
"""
from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 32000
N_FFT = 2048
HOP = 320            # 10 ms
N_MELS = 128
EXCERPT_S = 6.0
EXCERPT_FRAMES = int(EXCERPT_S * SR / HOP) + 1
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")


# --------------------------------------------------------------------------------------
# Audio front-end
# --------------------------------------------------------------------------------------
def load_audio(path, sr=SR, max_seconds=None, offset=0.0):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=max_seconds)
    return y.astype(np.float32)


_MEL = {}


def logmel(y, sr=SR):
    import librosa
    if sr not in _MEL:
        _MEL[sr] = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=20, fmax=sr / 2)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True)) ** 2
    M = _MEL[sr] @ S
    return np.log(M + 1e-8).astype(np.float32)  # (n_mels, frames)


def augment_wave(y, rng):
    """Robustness augmentation: gain, EQ tilt, band-limiting, resampling, noise."""
    import scipy.signal as sps
    if rng.random() < 0.5:
        y = y * rng.uniform(0.3, 1.0)
    if rng.random() < 0.3:  # band-limit like low-bitrate encoders (16 kHz / 11 kHz cut-offs)
        cutoff = rng.choice([11000, 15000, 16000])
        b, a = sps.butter(6, cutoff / (SR / 2), "low")
        y = sps.lfilter(b, a, y).astype(np.float32)
    if rng.random() < 0.2:  # resample through a lower rate
        lo = int(rng.choice([16000, 22050, 24000]))
        y = sps.resample_poly(sps.resample_poly(y, lo, SR), SR, lo).astype(np.float32)[: len(y)]
    if rng.random() < 0.2:  # gentle EQ tilt
        b, a = sps.butter(1, rng.uniform(300, 3000) / (SR / 2), rng.choice(["low", "high"]))
        y = (0.7 * y + 0.3 * sps.lfilter(b, a, y)).astype(np.float32)
    if rng.random() < 0.2:
        y = y + rng.normal(0, rng.uniform(1e-4, 3e-3), len(y)).astype(np.float32)
    return y


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Sequential() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        return F.gelu(self.b2(self.c2(h)) + self.skip(x))


class AudioDetector(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.GELU())
        self.body = nn.Sequential(ResBlock(w, w, 1), ResBlock(w, 2 * w, 2), ResBlock(2 * w, 4 * w, 2),
                                  ResBlock(4 * w, 8 * w, 2), ResBlock(8 * w, 8 * w, 2))
        self.att = nn.Conv1d(8 * w, 1, 1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(16 * w, 1))

    def forward(self, x):  # x: (B, n_mels, frames)
        h = self.body(self.stem(self.norm(x.unsqueeze(1))))   # (B, C, F', T')
        h = h.mean(2)                                          # pool frequency -> (B, C, T')
        a = torch.softmax(self.att(h), -1)
        pooled = torch.cat([(h * a).sum(-1), h.amax(-1)], 1)
        return self.head(pooled).squeeze(-1)


# --------------------------------------------------------------------------------------
# Data preparation: cache log-mel excerpts per track
# --------------------------------------------------------------------------------------
def list_tracks(root):
    items = []
    for label_name, label in (("real", 0), ("ai", 1)):
        base = os.path.join(root, label_name)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            rel = os.path.relpath(dirpath, base)
            source = rel.split(os.sep)[0] if rel != "." else label_name
            for f in files:
                if f.lower().endswith(AUDIO_EXT):
                    items.append((os.path.join(dirpath, f), label, f"{label_name}/{source}"))
    return sorted(items)


def _prepare_one(args):
    path, label, source, cache_dir, n_excerpts, augment, seed = args
    key = f"{zlib.crc32(path.encode()):08x}"
    out = os.path.join(cache_dir, key + ".npy")
    if os.path.exists(out):
        return dict(key=key, path=path, label=label, source=source)
    rng = np.random.default_rng(seed)
    try:
        import librosa
        dur = librosa.get_duration(path=path)
        starts = np.linspace(0, max(dur - EXCERPT_S, 0), n_excerpts) if dur > EXCERPT_S else [0.0]
        mels = []
        for s in starts:
            y = load_audio(path, offset=float(s), max_seconds=EXCERPT_S)
            if len(y) < SR * 2:
                continue
            y = np.pad(y, (0, max(0, int(EXCERPT_S * SR) - len(y))))
            if augment and rng.random() < 0.5:
                y = augment_wave(y, rng)
            mels.append(logmel(y)[:, :EXCERPT_FRAMES])
        if not mels:
            return None
        np.save(out, np.stack(mels).astype(np.float16))
        return dict(key=key, path=path, label=label, source=source)
    except Exception as e:  # noqa: BLE001
        print(f"skip {path}: {e}")
        return None


def prepare(root="data/audio", cache_dir="data/audio_cache", n_excerpts=8, workers=4, augment=True):
    os.makedirs(cache_dir, exist_ok=True)
    tracks = list_tracks(root)
    if not tracks:
        raise SystemExit(f"No audio found under {root}/real and {root}/ai")
    jobs = [(p, l, s, cache_dir, n_excerpts, augment, i) for i, (p, l, s) in enumerate(tracks)]
    with ProcessPoolExecutor(workers) as ex:
        index = [r for r in ex.map(_prepare_one, jobs, chunksize=4) if r is not None]
    with open(os.path.join(cache_dir, "index.json"), "w") as f:
        json.dump(index, f)
    by = {}
    for r in index:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"cached {len(index)} tracks:", json.dumps(by, indent=1))
    return index


class ExcerptDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, items, train=True):
        self.cache_dir, self.items, self.train = cache_dir, items, train
        self.rows = []
        for i, it in enumerate(items):
            n = np.load(os.path.join(cache_dir, it["key"] + ".npy"), mmap_mode="r").shape[0]
            self.rows += [(i, k) for k in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        i, k = self.rows[idx]
        m = np.load(os.path.join(self.cache_dir, self.items[i]["key"] + ".npy"), mmap_mode="r")[k]
        m = torch.from_numpy(np.array(m, dtype=np.float32))
        if m.shape[1] < EXCERPT_FRAMES:
            m = F.pad(m, (0, EXCERPT_FRAMES - m.shape[1]), value=float(m.min()))
        if self.train:  # SpecAugment-style masking
            f0 = np.random.randint(0, N_MELS - 12)
            m[f0:f0 + np.random.randint(0, 12)] = m.mean()
            t0 = np.random.randint(0, EXCERPT_FRAMES - 60)
            m[:, t0:t0 + np.random.randint(0, 60)] = m.mean()
        return m, torch.tensor(float(self.items[i]["label"])), i


def _split_items(index, holdout_source=None, seed=0):
    tr, va, te = [], [], []
    for it in index:
        if holdout_source and it["source"].endswith("/" + holdout_source):
            te.append(it)
            continue
        h = zlib.crc32((it["path"] + str(seed)).encode()) % 100
        (te if h < 10 else va if h < 20 else tr).append(it)
    return tr, va, te


@torch.no_grad()
def _track_scores(model, loader, n_items, device):
    model.eval()
    s, c = np.zeros(n_items), np.zeros(n_items)
    for x, _, i in loader:
        logit = model(x.to(device)).float().cpu().numpy()
        np.add.at(s, i.numpy(), logit)
        np.add.at(c, i.numpy(), 1)
    return s / np.maximum(c, 1)


def _metrics(items, logits):
    from sklearn.metrics import roc_auc_score
    y = np.array([it["label"] for it in items])
    p = 1 / (1 + np.exp(-logits))
    out = dict(n=len(items), acc=float(np.mean((p > 0.5) == y)))
    if len(set(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    per = {}
    for src in sorted({it["source"] for it in items}):
        sel = np.array([it["source"] == src for it in items])
        per[src] = dict(n=int(sel.sum()), acc=float(np.mean((p[sel] > 0.5) == y[sel])))
    out["per_source"] = per
    return out


def train(cache_dir="data/audio_cache", out_path="models/audio_detector.pt", epochs=20, batch_size=32,
          lr=1e-3, holdout_source=None, width=32, workers=4, patience=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(cache_dir, "index.json")) as f:
        index = json.load(f)
    tr, va, te = _split_items(index, holdout_source)
    if not tr or not va:
        raise SystemExit("Not enough data to split; add more tracks.")
    n_ai = sum(it["label"] for it in tr)
    pos_weight = torch.tensor((len(tr) - n_ai) / max(n_ai, 1), device=device)
    dl = lambda items, train_: torch.utils.data.DataLoader(
        ExcerptDataset(cache_dir, items, train_), batch_size=batch_size, shuffle=train_,
        num_workers=workers, pin_memory=device.type == "cuda", drop_last=train_)
    trl, val = dl(tr, True), dl(va, False)
    model = AudioDetector(width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(trl))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best, bad = -1.0, 0
    print(f"device={device} train={len(tr)} val={len(va)} test={len(te)} tracks "
          f"(holdout={holdout_source})")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y, _ in trl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y, pos_weight=pos_weight)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        m = _metrics(va, _track_scores(model, val, len(va), device))
        score = m.get("auc", m["acc"])
        print(f"epoch {ep+1}: loss={tot/len(trl):.4f} val acc={m['acc']:.3f} auc={m.get('auc', float('nan')):.3f}",
              flush=True)
        if score > best:
            best, bad = score, 0
            torch.save(dict(state_dict=model.state_dict(), width=width, sr=SR, val=m,
                            holdout=holdout_source), out_path)
        else:
            bad += 1
            if bad >= patience:
                break
    ck = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    if te:
        tm = _metrics(te, _track_scores(model, dl(te, False), len(te), device))
        print("TEST:", json.dumps(tm, indent=1))
        ck["test"] = tm
        torch.save(ck, out_path)
    return out_path


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
_AUD = {}


def load_audio_detector(path="models/audio_detector.pt"):
    if path not in _AUD:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = AudioDetector(ck.get("width", 32))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        _AUD[path] = (m, ck)
    return _AUD[path]


@torch.no_grad()
def detect_audio(path, detector_path="models/audio_detector.pt", max_excerpts=30):
    model, ck = load_audio_detector(detector_path)
    y = load_audio(path)
    dur = len(y) / SR
    n = int(EXCERPT_S * SR)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    starts = np.linspace(0, len(y) - n, min(max_excerpts, max(1, int(dur // 3)))).astype(int)
    mels = np.stack([logmel(y[s:s + n])[:, :EXCERPT_FRAMES] for s in starts])
    logits = model(torch.from_numpy(mels)).numpy()
    p = float(1 / (1 + np.exp(-logits.mean())))
    flagged = float(np.mean(logits > 0))
    notes = []
    if dur < 10:
        notes.append("Clip shorter than 10 s - low confidence.")
    if 0.3 < p < 0.7:
        notes.append("Score is in the uncertain zone.")
    notes.append("Only reliable for generators represented in the training data; "
                 "a probability is evidence, not proof.")
    verdict = "ai" if p >= 0.5 else "human"
    return dict(file=os.path.basename(path), type="audio", duration_s=round(dur, 1), p_ai=p,
                verdict=verdict,
                verdict_text="likely AI-generated audio" if verdict == "ai" else "likely human-made audio",
                excerpts=len(starts), excerpts_flagged=flagged, notes=notes)
