"""
File-level features describing *how* a MIDI file was produced.

Design rules
  * No container/format cues (track count, tick resolution, tempo-event count,
    controller presence, meta text...). Those are trivial to fake and would let a
    classifier learn dataset quirks instead of musical evidence.
  * Each feature captures a known difference between human playing and machine output:
      - grid alignment            programmed MIDI sits exactly on a grid
      - deviation structure       humans drift smoothly (autocorrelated), DAW
                                  'humanise' adds independent noise
      - chord asynchrony shape    pianists lead the melody; random jitter does not
      - velocity structure        humans shape dynamics continuously and voice chords
      - duration / legato         step-entered notes have grid-length durations
"""
from __future__ import annotations

import numpy as np

from ..features import BeatGrid

FEATURE_NAMES = [
    "n_notes_log", "drum_frac",
    "grid_1ms", "grid_4ms", "grid_10ms",
    "dev_std_rel", "dev_kurt", "dev_ac1", "dev_ac2", "dev_absmean_ms",
    "ioi_equal_1ms", "ioi_ratio_simple",
    "chord_exact_frac", "chord_spread_ms", "chord_lead_corr",
    "vel_unique", "vel_entropy", "vel_mode_frac", "vel_round_frac", "vel_std", "vel_ac1",
    "vel_step_zero", "chord_vel_pitch_corr", "chord_vel_std",
    "dur_grid_1ms", "dur_entropy", "legato_exact",
    "drum_dev_ac1", "drum_grid_4ms", "drum_vel_unique",
]


def _grid(pm):
    beats = pm.get_beats()
    end = pm.get_end_time() + 1
    if len(beats) < 2:
        beats = np.arange(0, end + 1, 0.5)
    elif beats[-1] < end:
        p = beats[-1] - beats[-2]
        beats = np.concatenate([beats, beats[-1] + p * np.arange(1, int((end - beats[-1]) / p) + 2)])
    return BeatGrid(beats)


def _ac(x, lag):
    if len(x) <= lag + 2 or np.std(x) < 1e-9:
        return 0.0
    a, b = x[:-lag] - x.mean(), x[lag:] - x.mean()
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def _corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _entropy(values, bins):
    h, _ = np.histogram(values, bins=bins)
    p = h[h > 0] / max(h.sum(), 1)
    return float(-(p * np.log2(p)).sum())


def _grid_dev(t, grid):
    """Signed distance (s) to nearest 1/12- or 1/8-beat grid point, and that grid step."""
    b = grid.to_beats(t)
    per = grid.beat_period(t)
    best = np.full(len(t), np.inf)
    signed = np.zeros(len(t))
    step = np.zeros(len(t))
    for div in (12, 8, 4):
        d = (b * div - np.round(b * div)) / div * per
        better = np.abs(d) < np.abs(best) - 1e-9
        best = np.where(better, d, best)
        step = np.where(better, per / div, step)
    signed = best
    return signed, step


def _timing_block(on, grid):
    if len(on) < 4:
        return dict(g1=1.0, g4=1.0, g10=1.0, std_rel=0, kurt=0, ac1=0, ac2=0, absmean=0)
    dev, _ = _grid_dev(on, grid)
    # deviation relative to the 16th-note grid, for structure statistics
    b = grid.to_beats(on)
    per = grid.beat_period(on)
    d16 = (b * 4 - np.round(b * 4)) / 4 * per
    # one value per onset group so chords don't dominate autocorrelation
    keep = np.concatenate([[True], np.diff(on) > 0.03])
    d = d16[keep]
    sd = np.std(d)
    kurt = float(np.mean((d - d.mean()) ** 4) / (sd ** 4 + 1e-12)) if sd > 1e-6 else 0.0
    return dict(
        g1=float(np.mean(np.abs(dev) < 0.001)),
        g4=float(np.mean(np.abs(dev) < 0.004)),
        g10=float(np.mean(np.abs(dev) < 0.010)),
        std_rel=float(sd / (np.median(per) / 4)),
        kurt=min(kurt, 20.0),
        ac1=_ac(d, 1), ac2=_ac(d, 2),
        absmean=float(np.mean(np.abs(d)) * 1000),
    )


def extract(pm):
    pitched = [(n.start, n.end, n.pitch, n.velocity) for i in pm.instruments if not i.is_drum for n in i.notes]
    drums = [(n.start, n.end, n.pitch, n.velocity) for i in pm.instruments if i.is_drum for n in i.notes]
    grid = _grid(pm)
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    total = len(pitched) + len(drums)
    f["n_notes_log"] = np.log1p(total)
    f["drum_frac"] = len(drums) / max(total, 1)

    main = pitched if len(pitched) >= 8 else drums
    if len(main) < 8:
        return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)
    a = np.array(sorted(main))
    on, off, pitch, vel = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    t = _timing_block(on, grid)
    f.update(grid_1ms=t["g1"], grid_4ms=t["g4"], grid_10ms=t["g10"], dev_std_rel=t["std_rel"],
             dev_kurt=t["kurt"], dev_ac1=t["ac1"], dev_ac2=t["ac2"], dev_absmean_ms=t["absmean"])

    # IOI structure (tempo-map independent)
    uon = on[np.concatenate([[True], np.diff(on) > 0.0005])]
    ioi = np.diff(uon)
    ioi = ioi[ioi > 0.03]
    if len(ioi) > 3:
        f["ioi_equal_1ms"] = float(np.mean(np.abs(np.diff(ioi)) < 0.001))
        r = np.log2(ioi[1:] / ioi[:-1])
        simple = np.log2([1, 2, 0.5, 3, 1 / 3, 1.5, 2 / 3, 4, 0.25])
        f["ioi_ratio_simple"] = float(np.mean(np.min(np.abs(r[:, None] - simple[None]), 1) < 0.01))

    # Chords: clusters of onsets within 40 ms
    gid = np.concatenate([[0], np.cumsum(np.diff(on) > 0.04)])
    sizes = np.bincount(gid)
    multi = np.where(sizes > 1)[0]
    if len(multi):
        exact, spreads, leads, vp, vstd = [], [], [], [], []
        for g in multi[:2000]:
            sel = gid == g
            o, p, v = on[sel], pitch[sel], vel[sel]
            exact.append(np.ptp(o) < 0.0005)
            spreads.append(np.std(o))
            if np.ptp(o) > 0.0005 and np.ptp(p) > 0:
                leads.append(_corr(p, o))
            if np.ptp(p) > 0 and np.ptp(v) > 0:
                vp.append(_corr(p, v))
            vstd.append(np.std(v))
        f["chord_exact_frac"] = float(np.mean(exact))
        f["chord_spread_ms"] = float(np.mean(spreads) * 1000)
        f["chord_lead_corr"] = float(np.mean(leads)) if leads else 0.0
        f["chord_vel_pitch_corr"] = float(np.mean(vp)) if vp else 0.0
        f["chord_vel_std"] = float(np.mean(vstd))

    # Velocity
    vals, counts = np.unique(vel, return_counts=True)
    f["vel_unique"] = len(vals) / 127.0
    f["vel_entropy"] = _entropy(vel, np.arange(0, 129))
    f["vel_mode_frac"] = counts.max() / len(vel)
    f["vel_round_frac"] = float(np.mean((vel % 5) == 0))
    f["vel_std"] = float(np.std(vel) / 127)
    f["vel_ac1"] = _ac(vel.astype(float), 1)
    f["vel_step_zero"] = float(np.mean(np.diff(vel) == 0))

    # Durations
    dur = np.maximum(off - on, 1e-4)
    per = grid.beat_period(on)
    best = np.full(len(dur), np.inf)
    for div in (12, 8, 4):
        step = per / div
        best = np.minimum(best, np.abs(dur - np.round(dur / step) * step))
    f["dur_grid_1ms"] = float(np.mean(best < 0.001))
    f["dur_entropy"] = _entropy(np.log2(dur), np.linspace(-8, 4, 97))
    nxt = np.searchsorted(on, off - 0.0005)
    nxt_on = on[np.minimum(nxt, len(on) - 1)]
    f["legato_exact"] = float(np.mean(np.abs(nxt_on - off) < 0.0005))

    if len(drums) >= 8:
        d = np.array(sorted(drums))
        td = _timing_block(d[:, 0], grid)
        f["drum_dev_ac1"] = td["ac1"]
        f["drum_grid_4ms"] = td["g4"]
        f["drum_vel_unique"] = len(np.unique(d[:, 3])) / 127.0
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)
