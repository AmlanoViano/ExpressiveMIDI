import os

import numpy as np
import pytest
import torch

from expressivemidi.detect.midi_detector import _snap_ticks, make_programmed, make_random
from expressivemidi.detect.midi_features import FEATURE_NAMES, extract
from conftest import make_song


def feats(pm):
    return dict(zip(FEATURE_NAMES, extract(pm)))


def test_midi_features_separate_programmed_and_jittered():
    rng = np.random.default_rng(1)
    q = feats(make_song())
    assert len(FEATURE_NAMES) == len(extract(make_song()))
    assert q["grid_1ms"] > 0.99 and q["chord_exact_frac"] > 0.99
    j = make_song()
    for inst in j.instruments:
        for n in inst.notes:
            d = rng.normal(0, 0.015); n.start += d; n.end += d
    j = feats(j)
    assert j["grid_1ms"] < 0.2
    assert abs(j["dev_ac1"]) < 0.35          # independent jitter -> little autocorrelation


def test_synthetic_variants_run():
    rng = np.random.default_rng(0)
    for fn in (make_programmed, make_random):
        pm = _snap_ticks(fn(make_song(), rng))
        assert np.isfinite(extract(pm)).all()


def test_audio_model_forward_and_logmel():
    from expressivemidi.detect.audio import EXCERPT_FRAMES, N_MELS, SR, AudioDetector, logmel
    y = np.random.default_rng(0).normal(0, 0.1, int(SR * 6)).astype(np.float32)
    m = logmel(y)[:, :EXCERPT_FRAMES]
    assert m.shape == (N_MELS, EXCERPT_FRAMES)
    model = AudioDetector(width=8).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(np.stack([m, m])))
    assert out.shape == (2,)


def test_audio_pipeline_end_to_end(tmp_path):
    """prepare -> train -> detect on tiny synthetic data (checks the plumbing, not accuracy)."""
    sf = pytest.importorskip("soundfile")
    from expressivemidi.detect import audio
    rng = np.random.default_rng(0)
    for label in ("real", "ai"):
        d = tmp_path / "audio" / label / "src"
        d.mkdir(parents=True)
        for k in range(6):
            t = np.arange(int(audio.SR * 7)) / audio.SR
            y = 0.3 * np.sin(2 * np.pi * (220 + 40 * k) * t) + rng.normal(0, 0.05, len(t))
            if label == "ai":  # crude "codec": hard band limit
                Y = np.fft.rfft(y); Y[len(Y) // 3:] = 0; y = np.fft.irfft(Y, len(y))
            sf.write(d / f"{k}.wav", y.astype(np.float32), audio.SR)
    cache = tmp_path / "cache"
    idx = audio.prepare(str(tmp_path / "audio"), str(cache), n_excerpts=2, workers=1, augment=False)
    assert len(idx) == 12
    out = tmp_path / "det.pt"
    audio.train(str(cache), str(out), epochs=1, batch_size=4, workers=0, width=8)
    assert os.path.exists(out)
    r = audio.detect_audio(str(tmp_path / "audio" / "ai" / "src" / "0.wav"), str(out))
    assert 0.0 <= r["p_ai"] <= 1.0 and r["verdict"] in ("ai", "human")
