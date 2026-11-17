"""Diagnostics for a trained audio detector (run after `train-audio-detector`).

    python scripts/eval_audio_detector.py --model models/audio_detector.pt \
        [--fma-tracks data/raw/fma_metadata/tracks.csv] [--external real:data/audio_external/jamendo ...]

1. Test split (same split as training): per-source accuracy, and accuracy by excerpt
   position (excerpt 0 = start of the file) to check the model is not keying on intros.
2. FMA test tracks by genre (needs fma_metadata/tracks.csv): false-alarm rate per genre.
3. External folders never used in training, labelled on the command line
   (`real:<dir>` or `ai:<dir>`): track-level accuracy with the normal inference path.
Nothing here feeds back into training; it is for reporting.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressivemidi.detect import audio  # noqa: E402


@torch.no_grad()
def excerpt_logits(model, cache_dir, items, device, batch_size=64):
    ds = audio.ExcerptDataset(cache_dir, items, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)
    out = []
    for x, _, _ in dl:
        out.append(model(x.to(device)).float().cpu().numpy())
    rows = np.array(ds.rows)
    return rows[:, 0], rows[:, 1], np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/audio_detector.pt")
    ap.add_argument("--cache", default="data/audio_cache")
    ap.add_argument("--fma-tracks", default=None)
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.model, map_location=device, weights_only=False)
    model = audio.AudioDetector(ck.get("width", 32)).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    report = dict(model=a.model, holdout=ck.get("holdout"), stored_test=ck.get("test"))

    index = json.load(open(os.path.join(a.cache, "index.json")))
    _, _, te = audio._split_items(index, ck.get("holdout"))
    ti, pos, lg = excerpt_logits(model, a.cache, te, device)
    y = np.array([te[i]["label"] for i in ti])
    report["excerpt_acc_by_position"] = {int(k): float(np.mean((lg[pos == k] > 0) == y[pos == k]))
                                         for k in sorted(set(pos.tolist()))}
    # track-level: excerpt 0 removed, to confirm the result does not depend on intros
    s, c = np.zeros(len(te)), np.zeros(len(te))
    np.add.at(s, ti[pos > 0], lg[pos > 0])
    np.add.at(c, ti[pos > 0], 1)
    keep = c > 0
    yt = np.array([it["label"] for it in te])
    report["track_acc_without_first_excerpt"] = float(np.mean(((s[keep] / c[keep]) > 0) == yt[keep]))

    if a.fma_tracks:
        import pandas as pd
        tracks = pd.read_csv(a.fma_tracks, index_col=0, header=[0, 1])
        genre = tracks[("track", "genre_top")]
        s, c = np.zeros(len(te)), np.zeros(len(te))
        np.add.at(s, ti, lg)
        np.add.at(c, ti, 1)
        per = {}
        for i, it in enumerate(te):
            if it["source"] != "real/fma":
                continue
            tid = int(os.path.splitext(os.path.basename(it["path"]))[0])
            g = genre.get(tid, "unknown")
            g = "unknown" if not isinstance(g, str) else g
            per.setdefault(g, []).append(s[i] / c[i] > 0)
        report["fma_false_alarm_by_genre"] = {g: dict(n=len(v), false_alarm=float(np.mean(v)))
                                              for g, v in sorted(per.items())}

    for spec in a.external:
        lab, d = spec.split(":", 1)
        files = sorted(os.path.join(r, f) for r, _, ff in os.walk(d) for f in ff
                       if f.lower().endswith(audio.AUDIO_EXT) and not f.startswith("."))
        ps, failed = [], 0
        for f in files:
            try:
                ps.append(audio.detect_audio(f, a.model)["p_ai"])
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("skip", f, e)
        ps = np.array(ps)
        target = 1 if lab == "ai" else 0
        report.setdefault("external", {})[d] = dict(
            label=lab, n=len(ps), failed=failed, acc=float(np.mean((ps >= 0.5) == target)),
            median_p_ai=float(np.median(ps)) if len(ps) else None)

    print(json.dumps(report, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
