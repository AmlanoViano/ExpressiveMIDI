"""
Build a training set from ASAP (Aligned Scores And Performances, Foscarin et al. 2020).

ASAP pairs 1,067 MAESTRO piano performances with their quantised score MIDI and with
beat/downbeat annotations for both. That gives what the original ExpressiveMIDI
pipeline was missing: the *score* the pianist was reading. Expressive deviations are
only meaningful relative to that score.

For every score note we find the performed note (same pitch, nearest to where the
beat alignment says it should be), and record:

    tempo   log(performed beat length / notated beat length), minus the piece median
            -> the rubato curve
    micro   performed onset minus the beat-interpolated onset, in ms
            -> chord asynchrony, melody lead, local push/pull
    vel     (performed velocity - piece mean) / 127
    artic   log(performed duration / notated duration under the local tempo)
    pedal   sustain pedal down shortly after the onset (0/1)
    repedal sustain pedal released and re-pressed around the onset (0/1)

    git clone https://github.com/fosfrancesco/asap-dataset data/raw/asap-dataset
    python -m expressivemidi build-dataset --asap data/raw/asap-dataset
"""
from __future__ import annotations

import os
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pretty_midi

from ..features import BeatGrid, score_features

TARGET_NAMES = ["tempo", "micro", "vel", "artic", "pedal", "repedal"]
N_TARGETS = len(TARGET_NAMES)
REGRESSION_TARGETS = 4  # the first four are regression, the rest binary


def read_annotations(path):
    times, labels = [], []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            times.append(float(parts[0]))
            labels.append(parts[2].split(",")[0])
    return np.array(times), labels


def all_notes(midi):
    notes = [n for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    notes.sort(key=lambda n: (n.start, n.pitch))
    on = np.array([n.start for n in notes])
    off = np.array([n.end for n in notes])
    pitch = np.array([n.pitch for n in notes])
    vel = np.array([n.velocity for n in notes])
    return on, off, pitch, vel


def pedal_curve(midi, number=64):
    ccs = sorted((c.time, c.value) for inst in midi.instruments for c in inst.control_changes
                 if c.number == number)
    if not ccs:
        return np.array([0.0]), np.array([0.0])
    t, v = zip(*ccs)
    return np.array(t), np.array(v, dtype=float)


def _value_at(times, values, t):
    idx = np.searchsorted(times, t, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], 0.0)


def match_notes(s_pitch, s_pred, p_on, p_pitch, tol):
    """Greedy nearest-neighbour matching per pitch. Returns perf index per score note (-1 = none)."""
    match = np.full(len(s_pitch), -1)
    for pitch in np.unique(s_pitch):
        s_idx = np.where(s_pitch == pitch)[0]
        p_idx = np.where(p_pitch == pitch)[0]
        if len(p_idx) == 0:
            continue
        p_times = p_on[p_idx]
        used = np.zeros(len(p_idx), dtype=bool)
        for i in s_idx[np.argsort(s_pred[s_idx], kind="stable")]:
            j = np.searchsorted(p_times, s_pred[i])
            best, best_d = -1, tol[i]
            for k in range(max(0, j - 3), min(len(p_idx), j + 4)):
                d = abs(p_times[k] - s_pred[i])
                if not used[k] and d < best_d:
                    best, best_d = k, d
            if best >= 0:
                used[best] = True
                match[i] = p_idx[best]
    return match


def process_performance(score_path, perf_path, score_ann, perf_ann, min_match=0.75):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pretty_midi.PrettyMIDI(score_path)
        perf = pretty_midi.PrettyMIDI(perf_path)
    s_beats, s_labels = read_annotations(score_ann)
    p_beats, _ = read_annotations(perf_ann)
    if len(s_beats) != len(p_beats) or len(s_beats) < 8:
        return None
    keep = np.concatenate([[True], (np.diff(s_beats) > 1e-3) & (np.diff(p_beats) > 1e-3)])
    s_beats, p_beats = s_beats[keep], p_beats[keep]
    s_labels = [l for l, k in zip(s_labels, keep) if k]
    downbeats = s_beats[[l.startswith("db") for l in s_labels]]

    s_on, s_off, s_pitch, _ = all_notes(score)
    p_on, p_off, p_pitch, p_vel = all_notes(perf)
    if len(s_on) < 32 or len(p_on) < 32:
        return None

    s_grid = BeatGrid(s_beats, downbeats)
    p_grid = BeatGrid(p_beats)
    s_on_b = s_grid.to_beats(s_on)
    s_off_b = s_grid.to_beats(s_off)
    pred_on = p_grid.to_seconds(s_on_b)
    pred_off = p_grid.to_seconds(s_off_b)
    p_period = p_grid.beat_period(pred_on)

    order = np.argsort(p_on, kind="stable")
    p_on, p_off, p_pitch, p_vel = p_on[order], p_off[order], p_pitch[order], p_vel[order]
    tol = np.clip(0.45 * p_period, 0.07, 0.4)
    m = match_notes(s_pitch, pred_on, p_on, p_pitch, tol)
    ok = m >= 0
    if ok.mean() < min_match:
        return None

    T = np.zeros((len(s_on), 6), dtype=np.float32)
    M = np.zeros((len(s_on), 6), dtype=np.float32)

    # tempo (rubato): per beat segment, defined for every score note
    seg = np.clip(np.floor(s_on_b).astype(int), 0, len(s_beats) - 2)
    ratio = np.log(np.diff(p_beats) / np.diff(s_beats))
    ratio = ratio - np.median(ratio)
    T[:, 0] = np.clip(ratio[seg], -0.8, 0.8)
    M[:, 0] = 1.0

    mi = m[ok]
    T[ok, 1] = np.clip((p_on[mi] - pred_on[ok]) * 1000.0, -150, 150)
    vel = p_vel[mi].astype(float)
    T[ok, 2] = np.clip((vel - vel.mean()) / 127.0, -0.5, 0.5)
    notated = np.maximum(pred_off[ok] - pred_on[ok], 0.02)
    T[ok, 3] = np.clip(np.log(np.maximum(p_off[mi] - p_on[mi], 0.01) / notated), -3.0, 1.5)
    M[ok, 1:4] = 1.0

    ped_t, ped_v = pedal_curve(perf)
    if len(ped_t) > 1:
        down_after = _value_at(ped_t, ped_v, p_on[mi] + 0.08) >= 64
        ups = ped_t[1:][(ped_v[:-1] >= 64) & (ped_v[1:] < 64)]
        repedal = np.array([np.any((ups > t - 0.12) & (ups < t + 0.12)) for t in p_on[mi]])
        T[ok, 4] = down_after
        T[ok, 5] = repedal & down_after
        M[ok, 4:6] = 1.0

    X = score_features(s_on, s_off, s_pitch, s_grid)
    return dict(X=X, Y=T, M=M, match_rate=float(ok.mean()))


def _job(args):
    asap_dir, row = args
    j = lambda p: os.path.join(asap_dir, p)
    try:
        out = process_performance(j(row["midi_score"]), j(row["midi_performance"]),
                                  j(row["midi_score_annotations"]), j(row["performance_annotations"]))
    except Exception as e:  # noqa: BLE001 - keep going on malformed files
        print(f"skip {row['midi_performance']}: {e}")
        return None
    if out is not None:
        out["folder"] = row["folder"]
        out["perf"] = row["midi_performance"]
    return out


def split_of(folder):
    """Deterministic split by *piece* so no piece appears in two splits."""
    h = zlib.crc32(folder.encode()) % 100
    return "test" if h < 8 else "validation" if h < 16 else "train"


def build_dataset(asap_dir, out_dir="data/processed", workers=4, limit=None):
    meta = pd.read_csv(os.path.join(asap_dir, "metadata.csv"))
    if limit:
        meta = meta.iloc[:limit]
    rows = [(asap_dir, r) for _, r in meta.iterrows()]
    with ProcessPoolExecutor(workers) as ex:
        results = [r for r in ex.map(_job, rows, chunksize=4) if r is not None]
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for split in ("train", "validation", "test"):
        part = [r for r in results if split_of(r["folder"]) == split]
        lengths = np.array([len(r["X"]) for r in part])
        np.savez_compressed(
            os.path.join(out_dir, f"asap_{split}.npz"),
            X=np.concatenate([r["X"] for r in part]),
            Y=np.concatenate([r["Y"] for r in part]),
            M=np.concatenate([r["M"] for r in part]),
            lengths=lengths,
            folders=np.array([r["folder"] for r in part]),
            perfs=np.array([r["perf"] for r in part]),
        )
        stats[split] = (len(part), int(lengths.sum()))
        print(f"{split:>10}: {len(part)} performances, {lengths.sum():,} notes")
    rates = [r["match_rate"] for r in results]
    print(f"kept {len(results)}/{len(meta)} performances, mean note match rate {np.mean(rates):.1%}")
    return stats
