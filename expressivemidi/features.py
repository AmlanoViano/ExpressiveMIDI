"""
Score-level note features.

Every feature here is computed from information that exists in a *quantised score*
(pitches, notated onsets/durations and a beat grid). Nothing is derived from a
performance, so the features are identical at training time (ASAP score MIDI) and at
inference time (any quantised MIDI a user exports from a DAW / notation program).
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "pitch_c",          # (pitch - 60) / 24
    "pc_sin", "pc_cos",  # pitch class on the circle
    "log_dur_beats",
    "log_ioi_next",     # time to next distinct onset (beats)
    "log_ioi_prev",
    "bar_sin", "bar_cos",  # position within bar
    "metric_strength",
    "beats_per_bar",
    "chord_size",
    "chord_top", "chord_bottom", "chord_rank",
    "pitch_rel_local",  # pitch relative to local (+-2 beats) mean
    "top_interval",     # interval between this group's top note and the previous group's
    "log_density",      # onsets per beat around the note
    "rest_before",      # silence before the onset (beats, log1p)
    "rest_after",       # silence after the note group (beats, log1p)
    "piece_pos",        # 0..1
    "legato",           # notated duration / time to next onset
    "log_dur_secs",     # notated duration in seconds (tempo information)
]
N_FEATURES = len(FEATURE_NAMES)

CHORD_TOL_BEATS = 0.03


class BeatGrid:
    """Map seconds <-> fractional beat index, extrapolating linearly at both ends."""

    def __init__(self, beats, downbeats=None):
        beats = np.asarray(beats, dtype=np.float64)
        beats = beats[np.concatenate([[True], np.diff(beats) > 1e-4])]
        if len(beats) < 2:
            beats = np.array([0.0, 0.5]) if len(beats) == 0 else np.array([beats[0], beats[0] + 0.5])
        self.beats = beats
        if downbeats is None or len(downbeats) == 0:
            self.downbeat_idx = np.arange(0, len(beats) + 4, 4, dtype=np.float64)
        else:
            self.downbeat_idx = np.unique(np.round(self.to_beats(np.asarray(downbeats)), 3))

    def to_beats(self, t):
        t = np.asarray(t, dtype=np.float64)
        b = self.beats
        idx = np.interp(t, b, np.arange(len(b), dtype=np.float64))
        p0, p1 = b[1] - b[0], b[-1] - b[-2]
        before, after = t < b[0], t > b[-1]
        idx = np.where(before, (t - b[0]) / p0, idx)
        idx = np.where(after, len(b) - 1 + (t - b[-1]) / p1, idx)
        return idx

    def to_seconds(self, beat_idx, beats=None):
        b = self.beats if beats is None else np.asarray(beats)
        x = np.asarray(beat_idx, dtype=np.float64)
        t = np.interp(x, np.arange(len(b), dtype=np.float64), b)
        p0, p1 = b[1] - b[0], b[-1] - b[-2]
        t = np.where(x < 0, b[0] + x * p0, t)
        t = np.where(x > len(b) - 1, b[-1] + (x - (len(b) - 1)) * p1, t)
        return t

    def beat_period(self, t):
        """Seconds per beat at time t."""
        b = self.beats
        k = np.clip(np.floor(self.to_beats(t)).astype(int), 0, len(b) - 2)
        return b[k + 1] - b[k]

    def bar_position(self, beat_idx):
        """Return (fraction within bar 0..1, beats in bar)."""
        db = self.downbeat_idx
        if len(db) < 2:
            db = np.arange(0, beat_idx.max() + 8, 4, dtype=np.float64)
        k = np.searchsorted(db, beat_idx + 1e-6, side="right") - 1
        lengths = np.diff(db)
        lengths = np.concatenate([lengths, [lengths[-1]]])
        k_c = np.clip(k, 0, len(db) - 1)
        bar_len = np.where(k < 0, lengths[0], lengths[k_c])
        start = np.where(k < 0, db[0] - lengths[0] * np.ceil((db[0] - beat_idx) / lengths[0]), db[k_c])
        frac = (beat_idx - start) / np.maximum(bar_len, 1e-6)
        return np.clip(frac, 0, 0.9999), bar_len


def onset_groups(onset_beats, tol=CHORD_TOL_BEATS):
    """Group index for each note (notes must be sorted by onset)."""
    if len(onset_beats) == 0:
        return np.zeros(0, dtype=int)
    new = np.concatenate([[True], np.diff(onset_beats) > tol])
    return np.cumsum(new) - 1


def _metric_strength(bar_frac, beat_idx):
    frac = beat_idx - np.floor(beat_idx)
    near = lambda x, v, tol=0.03: np.abs(x - v) < tol
    s = np.full(len(beat_idx), 0.1)
    s = np.where(near(frac * 3, np.round(frac * 3), 0.06), 0.2, s)    # triplet grid
    s = np.where(near(frac * 4, np.round(frac * 4), 0.08), 0.2, s)    # 16th grid
    s = np.where(near(frac, 0.5), 0.35, s)                           # 8th off-beat
    s = np.where(near(frac, 0) | near(frac, 1), 0.6, s)              # on the beat
    s = np.where(bar_frac < 0.01, 1.0, s)                           # downbeat
    return s


def score_features(onsets, offsets, pitches, grid: BeatGrid):
    """
    onsets/offsets in seconds (notated), pitches as MIDI numbers.
    Notes must be sorted by (onset, pitch). Returns (N, N_FEATURES) float32.
    """
    onsets = np.asarray(onsets, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    pitches = np.asarray(pitches, dtype=np.float64)
    n = len(onsets)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    on_b = grid.to_beats(onsets)
    off_b = grid.to_beats(offsets)
    dur_b = np.maximum(off_b - on_b, 1e-3)

    g = onset_groups(on_b)
    n_groups = g[-1] + 1
    g_onset = _group_min(on_b, g, n_groups)

    g_next = np.concatenate([g_onset[1:], [g_onset[-1] + 4.0]])
    g_prev = np.concatenate([[g_onset[0] - 4.0], g_onset[:-1]])
    ioi_next = np.clip(g_next[g] - on_b, 1e-3, 16)
    ioi_prev = np.clip(on_b - g_prev[g], 1e-3, 16)

    bar_frac, bar_len = grid.bar_position(on_b)

    # Chord structure
    g_size = np.bincount(g, minlength=n_groups)
    g_top = np.full(n_groups, -1.0)
    g_bot = np.full(n_groups, 999.0)
    np.maximum.at(g_top, g, pitches)
    np.minimum.at(g_bot, g, pitches)
    size = g_size[g]
    # rank within chord by pitch (0 = lowest, 1 = highest)
    order = np.lexsort((pitches, g))
    pos_in_group = np.empty(n)
    starts = np.concatenate([[0], np.cumsum(g_size)[:-1]])
    pos_in_group[order] = np.arange(n) - np.repeat(starts, g_size)
    rank = np.where(size > 1, pos_in_group / np.maximum(size - 1, 1), 0.5)
    top_int = np.concatenate([[0.0], np.diff(g_top)])[g]

    # Local windows (+-2 beats) using sorted onsets
    lo = np.searchsorted(on_b, on_b - 2.0 - 1e-6, side="left")
    hi = np.searchsorted(on_b, on_b + 2.0 + 1e-6, side="right")
    csum = np.concatenate([[0.0], np.cumsum(pitches)])
    local_mean = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
    glo = np.searchsorted(g_onset, on_b - 2.0 - 1e-6, side="left")
    ghi = np.searchsorted(g_onset, on_b + 2.0 + 1e-6, side="right")
    density = (ghi - glo) / 4.0

    # Rests: running max of offsets of all earlier groups
    g_maxoff = np.full(n_groups, -np.inf)
    np.maximum.at(g_maxoff, g, off_b)
    run_max = np.maximum.accumulate(g_maxoff)
    prev_max = np.concatenate([[-np.inf], run_max[:-1]])
    rest_before = np.clip(g_onset - prev_max, 0, 16)
    rest_before[0] = 4.0
    rest_after = np.clip(g_next - run_max, 0, 16)
    rest_after[-1] = 4.0

    span = max(on_b[-1] - on_b[0], 1e-3)
    beat_sec = grid.beat_period(onsets)

    feats = np.stack([
        (pitches - 60) / 24.0,
        np.sin(2 * np.pi * (pitches % 12) / 12),
        np.cos(2 * np.pi * (pitches % 12) / 12),
        np.log2(dur_b),
        np.log2(ioi_next),
        np.log2(ioi_prev),
        np.sin(2 * np.pi * bar_frac),
        np.cos(2 * np.pi * bar_frac),
        _metric_strength(bar_frac, on_b),
        bar_len / 4.0,
        np.minimum(size, 8) / 4.0,
        (pitches == g_top[g]).astype(float),
        (pitches == g_bot[g]).astype(float),
        rank,
        (pitches - local_mean) / 12.0,
        np.clip(top_int, -24, 24) / 12.0,
        np.log2(density + 0.25),
        np.log1p(rest_before[g]),
        np.log1p(rest_after[g]),
        (on_b - on_b[0]) / span,
        np.clip(dur_b / ioi_next, 0, 4),
        np.log2(np.maximum(dur_b * beat_sec, 1e-3)),
    ], axis=1)
    return np.nan_to_num(feats).astype(np.float32)


def _group_min(values, g, n_groups):
    out = np.full(n_groups, np.inf)
    np.minimum.at(out, g, values)
    return out
