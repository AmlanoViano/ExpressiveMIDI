import os
import sys

import numpy as np
import pretty_midi
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expressivemidi.data.asap import TARGET_NAMES  # noqa: E402
from expressivemidi.features import FEATURE_NAMES  # noqa: E402
from expressivemidi.models.humaniser_net import HumaniserNet  # noqa: E402


def make_song(tempo=100.0, bars=8, drums=True, ts=(4, 4)):
    """Quantised multi-track song: piano (chords + melody), bass, optional drums."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(ts[0], ts[1], 0))
    beat = 60.0 / tempo
    piano = pretty_midi.Instrument(program=0, name="Piano")
    bass = pretty_midi.Instrument(program=33, name="Bass")
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for bar in range(bars):
        t0 = bar * ts[0] * beat
        for p in (60, 64, 67):
            piano.notes.append(pretty_midi.Note(80, p, t0, t0 + 2 * beat))
        for k in range(ts[0] * 2):
            t = t0 + k * beat / 2
            piano.notes.append(pretty_midi.Note(80, 72 + (k * 2) % 7, t, t + beat / 2))
        bass.notes.append(pretty_midi.Note(90, 36, t0, t0 + beat))
        for k in range(ts[0] * 2):
            t = t0 + k * beat / 2
            kit.notes.append(pretty_midi.Note(90, 42, t, t + 0.05))
            if k % 4 == 0:
                kit.notes.append(pretty_midi.Note(100, 36, t, t + 0.05))
    pm.instruments += [piano, bass] + ([kit] if drums else [])
    return pm


@pytest.fixture
def song():
    return make_song()


@pytest.fixture
def fake_model(tmp_path):
    """Small randomly initialised humaniser with a valid checkpoint layout."""
    torch.manual_seed(0)
    model = HumaniserNet(len(FEATURE_NAMES), len(TARGET_NAMES), d_model=32, nhead=2, num_layers=1)
    ck = dict(state_dict=model.state_dict(), config=model.config, feature_names=FEATURE_NAMES,
              target_names=TARGET_NAMES, x_mean=[0.0] * len(FEATURE_NAMES), x_std=[1.0] * len(FEATURE_NAMES),
              y_mean=[0.0, 0.0, 0.0, 0.0], y_std=[0.1, 30.0, 0.1, 0.4], seq_len=32)
    path = tmp_path / "humaniser.pt"
    torch.save(ck, path)
    model.eval()
    return model, ck, str(path)
