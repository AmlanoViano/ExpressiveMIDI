"""
Apply a trained humaniser to any MIDI file.

The file is edited in place (a deep copy), so tracks, programs, drums, tempo map,
time/key signatures, existing controllers and pitch bends are all preserved. Only
note timing, durations and velocities change (plus optional generated sustain pedal).

Pipeline per file
  1. Build the beat grid from the file's own tempo map / time signatures.
  2. Group pitched tracks into parts (all piano-family tracks form one part, any other
     instrument is its own part) and predict expression for every note.
  3. Rubato: per-beat tempo predictions are pooled, smoothed and turned into a time
     warp that is applied to *every* event (incl. drums & CCs) so tracks stay in sync.
  4. Per-note micro-timing, velocity and articulation are added on top.
  5. Drums get rule-based groove humanisation (the piano model knows nothing of drums).
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

import numpy as np
import pretty_midi
import torch

from . import __version__
from .features import BeatGrid, score_features
from .models.humaniser_net import HumaniserNet

TAG = "Humanised with ExpressiveMIDI"


@dataclass
class Settings:
    strength: float = 1.0       # global multiplier for everything below
    timing: float = 1.0         # micro-timing amount
    rubato: float | None = None  # None = auto (on for solo/no-drum files, off when drums present)
    velocity: float = 1.0
    articulation: float = 1.0
    pedal: str = "auto"          # piano parts only. auto = if no CC64 yet; on = replace existing; off
    drums: float = 1.0           # drum groove humanisation amount
    seed: int = 0
    tag: bool = True             # add a text meta-event saying the file was humanised


@dataclass
class Report:
    notes: int = 0
    parts: list = field(default_factory=list)
    rubato: float = 0.0
    mean_abs_shift_ms: float = 0.0
    velocity_change: float = 0.0
    pedal_events: int = 0
    drum_notes: int = 0
    warnings: list = field(default_factory=list)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = HumaniserNet(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def predict(model, ck, feats):
    """Sliding-window prediction with triangular cross-fade. Returns dict of per-note arrays."""
    n = len(feats)
    L = ck["seq_len"]
    x = (feats - np.array(ck["x_mean"], np.float32)) / np.array(ck["x_std"], np.float32)
    out = np.zeros((n, len(ck["target_names"])), np.float64)
    wsum = np.zeros(n)
    stride = L // 2
    starts = list(range(0, max(n - L, 0) + 1, stride))
    if n > L and starts[-1] != n - L:
        starts.append(n - L)
    w_full = np.minimum(np.arange(1, L + 1), np.arange(L, 0, -1)).astype(float)
    for s in starts:
        e = min(s + L, n)
        chunk = torch.from_numpy(x[s:e].astype(np.float32)).unsqueeze(0)
        pred = model(chunk).squeeze(0).numpy()
        w = w_full[: e - s] if n > L else np.ones(e - s)
        out[s:e] += pred * w[:, None]
        wsum[s:e] += w
    out /= wsum[:, None]
    r = len(ck["y_mean"])
    out[:, :r] = out[:, :r] * np.array(ck["y_std"]) + np.array(ck["y_mean"])
    res = {name: out[:, k] for k, name in enumerate(ck["target_names"])}
    for name in ck["target_names"][r:]:
        res[name] = 1 / (1 + np.exp(-res[name]))
    return res


def _grid_for(pm):
    beats = pm.get_beats()
    end = pm.get_end_time() + 2.0
    if len(beats) < 2:
        beats = np.arange(0, max(end, 2.0), 0.5)
    elif beats[-1] < end:  # extend so every event is inside the grid
        p = beats[-1] - beats[-2]
        beats = np.concatenate([beats, beats[-1] + p * np.arange(1, int((end - beats[-1]) / p) + 2)])
    try:
        downbeats = pm.get_downbeats()
    except Exception:  # noqa: BLE001
        downbeats = None
    return BeatGrid(beats, downbeats)


def quantisation_ratio(pm, grid=None, tol=0.004):
    """Fraction of pitched note onsets lying (within tol seconds) on a 1/12 or 1/8 beat grid."""
    grid = grid or _grid_for(pm)
    on = np.array([n.start for i in pm.instruments if not i.is_drum for n in i.notes])
    if len(on) == 0:
        on = np.array([n.start for i in pm.instruments for n in i.notes])
    if len(on) == 0:
        return 1.0
    b = grid.to_beats(on)
    period = grid.beat_period(on)
    best = np.full(len(on), np.inf)
    for div in (12, 8):
        d = np.abs(b * div - np.round(b * div)) / div * period
        best = np.minimum(best, d)
    return float(np.mean(best < tol))


def _parts(pm):
    piano = [i for i in pm.instruments if not i.is_drum and i.program < 8 and i.notes]
    others = [[i] for i in pm.instruments if not i.is_drum and i.program >= 8 and i.notes]
    return ([piano] if piano else []) + others


def _smooth(x, sigma):
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def humanise_midi(pm: pretty_midi.PrettyMIDI, model, ck, settings: Settings = Settings()):
    s = settings
    pm = copy.deepcopy(pm)
    rep = Report()
    rng = np.random.default_rng(s.seed)
    grid = _grid_for(pm)

    q = quantisation_ratio(pm, grid)
    if q < 0.5:
        rep.warnings.append(
            f"Only {q:.0%} of notes sit on a rhythmic grid - this file already looks performed/humanised. "
            "Humanising it again stacks extra deviations on top.")

    has_drums = any(i.is_drum and i.notes for i in pm.instruments)
    rubato = (0.0 if has_drums else 1.0) if s.rubato is None else s.rubato
    rubato *= s.strength
    rep.rubato = rubato

    # ---- predictions per part -------------------------------------------------------
    part_data = []
    for part in _parts(pm):
        notes = sorted((n for inst in part for n in inst.notes), key=lambda n: (n.start, n.pitch))
        on = np.array([n.start for n in notes])
        off = np.array([max(n.end, n.start + 1e-3) for n in notes])
        pitch = np.array([n.pitch for n in notes])
        pred = predict(model, ck, score_features(on, off, pitch, grid))
        part_data.append((part, notes, on, off, pred))
        rep.parts.append(f"{'piano' if part[0].program < 8 else pretty_midi.program_to_instrument_name(part[0].program)}"
                         f" ({len(notes)} notes)")

    # ---- rubato -> global time warp ---------------------------------------------------
    nseg = len(grid.beats) - 1
    if rubato > 0 and part_data:
        acc, cnt = np.zeros(nseg), np.zeros(nseg)
        for _, _, on, _, pred in part_data:
            seg = np.clip(np.floor(grid.to_beats(on)).astype(int), 0, nseg - 1)
            np.add.at(acc, seg, pred["tempo"])
            np.add.at(cnt, seg, 1)
        have = cnt > 0
        tempo = np.zeros(nseg)
        if have.any():
            idx = np.arange(nseg)
            tempo = np.interp(idx, idx[have], acc[have] / cnt[have])
        tempo = _smooth(tempo, 1.0) * rubato
        tempo = np.clip(tempo, -0.5, 0.5)
        periods = np.diff(grid.beats)
        scale = np.exp(tempo)
        scale /= (periods * scale).sum() / periods.sum()  # keep total duration
        new_beats = np.concatenate([[grid.beats[0]], grid.beats[0] + np.cumsum(periods * scale)])
        warp = lambda t: grid.to_seconds(grid.to_beats(t), beats=new_beats)
    else:
        warp = lambda t: np.asarray(t, dtype=np.float64)

    # Warp controllers / pitch bends of every track so they stay attached to the notes.
    for inst in pm.instruments:
        if inst.control_changes:
            t = warp([c.time for c in inst.control_changes])
            for c, tt in zip(inst.control_changes, t):
                c.time = float(max(tt, 0.0))
        if inst.pitch_bends:
            t = warp([b.time for b in inst.pitch_bends])
            for b, tt in zip(inst.pitch_bends, t):
                b.time = float(max(tt, 0.0))

    # ---- per-note expression -----------------------------------------------------------
    shifts, vchanges = [], []
    for part, notes, on, off, pred in part_data:
        is_piano = part[0].program < 8
        w_on, w_off = warp(on), warp(off)
        beat_sec = grid.beat_period(on)

        micro = pred["micro"] - np.median(pred["micro"])
        micro = micro * s.strength * s.timing / 1000.0
        # never let a note move past its neighbours' onsets (keeps rhythm readable)
        uniq = np.unique(np.round(w_on, 4))
        nxt = uniq[np.minimum(np.searchsorted(uniq, np.round(w_on, 4), side="right"), len(uniq) - 1)]
        prv = uniq[np.maximum(np.searchsorted(uniq, np.round(w_on, 4), side="left") - 1, 0)]
        room = np.minimum(np.where(nxt > w_on, nxt - w_on, np.inf), np.where(prv < w_on, w_on - prv, np.inf))
        lim = np.minimum(0.3 * np.minimum(room, beat_sec), 0.07)
        micro = np.clip(micro, -lim, lim)

        art = pred["artic"] - np.median(pred["artic"])
        art_amt = s.strength * s.articulation * (1.0 if is_piano else 0.5)
        dur_scale = np.clip(np.exp(art * art_amt), 0.45, 1.3)

        new_on = np.maximum(w_on + micro, 0.0)
        new_dur = np.maximum((w_off - w_on) * dur_scale, 0.03)

        base_vel = np.array([n.velocity for n in notes], dtype=float)
        dv = pred["vel"] * 127.0 * s.strength * s.velocity
        new_vel = np.clip(np.round(base_vel + dv), 1, 127)

        for k, n in enumerate(notes):
            shifts.append(abs(new_on[k] - n.start))
            vchanges.append(abs(new_vel[k] - n.velocity))
            n.start = float(new_on[k])
            n.end = float(new_on[k] + new_dur[k])
            n.velocity = int(new_vel[k])
        rep.notes += len(notes)

        # pedal
        has_cc64 = any(c.number == 64 for inst in part for c in inst.control_changes)
        want_pedal = is_piano and (s.pedal == "on" or (s.pedal == "auto" and not has_cc64))
        if want_pedal and s.strength > 0:
            if has_cc64:  # 'on' replaces the existing pedalling
                for inst in part:
                    inst.control_changes = [c for c in inst.control_changes if c.number != 64]
            rep.pedal_events += _add_pedal(part[0], new_on, new_on + new_dur, on, grid, pred)

    for inst in pm.instruments:
        _fix_overlaps(inst)

    # ---- drums ---------------------------------------------------------------------------
    for inst in pm.instruments:
        if inst.is_drum and inst.notes and s.drums > 0:
            rep.drum_notes += _humanise_drums(inst, grid, warp, s.strength * s.drums, rng)
        elif inst.is_drum and inst.notes and rubato > 0:
            for n in inst.notes:
                d = n.end - n.start
                n.start = float(warp([n.start])[0])
                n.end = n.start + d

    if s.tag:
        pm.text_events.append(pretty_midi.Text(f"{TAG} {__version__}", 0.0))
    rep.mean_abs_shift_ms = float(np.mean(shifts) * 1000) if shifts else 0.0
    rep.velocity_change = float(np.mean(vchanges)) if vchanges else 0.0
    return pm, rep


def _add_pedal(inst, new_on, new_off, score_on, grid, pred):
    """Turn per-note pedal probabilities into realistic CC64 events (legato pedalling)."""
    order = np.argsort(new_on, kind="stable")
    b = grid.to_beats(score_on[order])
    g = np.concatenate([[True], np.diff(b) > 0.03]).cumsum() - 1
    ng = g[-1] + 1
    t = np.full(ng, np.inf)
    np.minimum.at(t, g, new_on[order])
    p_down = np.bincount(g, pred["pedal"][order], ng) / np.bincount(g, minlength=ng)
    p_rep = np.bincount(g, pred["repedal"][order], ng) / np.bincount(g, minlength=ng)
    # median filter to avoid flickering
    p_down = np.array([np.median(p_down[max(0, i - 1): i + 2]) for i in range(ng)])
    events, state = [], False
    for k in range(ng):
        down = p_down[k] > 0.5
        if down and not state:
            events.append((t[k] + 0.04, 127))
            state = True
        elif down and state and p_rep[k] > 0.35:
            events.append((max(t[k] - 0.005, 0.0), 0))
            events.append((t[k] + 0.06, 127))
        elif not down and state:
            events.append((max(t[k] - 0.005, 0.0), 0))
            state = False
    if state:
        events.append((float(np.max(new_off)), 0))
    for time, val in events:
        inst.control_changes.append(pretty_midi.ControlChange(64, val, float(time)))
    inst.control_changes.sort(key=lambda c: c.time)
    return len(events)


def _fix_overlaps(inst):
    """Same-pitch notes must not overlap (most synths would cut or merge them)."""
    by_pitch = {}
    for n in inst.notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    for lst in by_pitch.values():
        lst.sort(key=lambda n: n.start)
        for a, b in zip(lst, lst[1:]):
            if a.end > b.start - 0.005:
                a.end = max(a.start + 0.01, b.start - 0.005)
    inst.notes.sort(key=lambda n: (n.start, n.pitch))


HIHATS = {42, 44, 46}
KICKS = {35, 36}
SNARES = {38, 40, 37}


def _humanise_drums(inst, grid, warp, amount, rng):
    """
    Groove humanisation for drums: correlated (AR(1)) timing drift instead of
    independent random jitter, consistent per-instrument feel (hats slightly ahead,
    snare slightly laid back) and metric velocity accents.
    """
    notes = sorted(inst.notes, key=lambda n: n.start)
    on = np.array([n.start for n in notes])
    b = grid.to_beats(on)
    frac = b - np.floor(b)
    drift = np.zeros(len(notes))
    e = rng.normal(0, 1, len(notes))
    for k in range(1, len(notes)):
        drift[k] = 0.6 * drift[k - 1] + 0.8 * e[k]
    bias = np.array([-0.003 if n.pitch in HIHATS else 0.004 if n.pitch in SNARES else 0.0 for n in notes])
    shift = (drift * 0.004 + bias) * amount
    accent = np.where(np.abs(frac) < 0.03, 6, np.where(np.abs(frac - 0.5) < 0.03, -2, -6))
    bar_frac, _ = grid.bar_position(b)
    accent = np.where(bar_frac < 0.01, 10, accent)
    accent = np.where([n.pitch in KICKS for n in notes], accent * 0.5, accent)
    dv = (accent * 0.6 + rng.normal(0, 4, len(notes))) * amount
    w = warp(on)
    for k, n in enumerate(notes):
        d = n.end - n.start
        n.start = float(max(w[k] + shift[k], 0.0))
        n.end = n.start + d
        n.velocity = int(np.clip(round(n.velocity + dv[k]), 1, 127))
    return len(notes)


def humanise_file(input_path, output_path, model_path="models/humaniser.pt", settings: Settings = Settings(),
                  model=None, ck=None):
    if model is None:
        model, ck = load_model(model_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = pretty_midi.PrettyMIDI(input_path)
    out, rep = humanise_midi(pm, model, ck, settings)
    out.write(output_path)
    return rep
