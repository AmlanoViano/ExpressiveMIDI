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


def test_audio_split_keeps_near_duplicates_together():
    from expressivemidi.detect.audio import _split_items
    index = []
    for i in range(300):  # SONICS-style siblings: two songs per prompt
        for k in (0, 1):
            index.append(dict(path=f"ai/suno/fake_{i:05d}_suno_{k}.mp3", label=1, source="ai/suno"))
    for i in range(300):  # FakeMusicCaps-style: same prompt file name under each generator
        for gen in ("musicgen", "audioldm2"):
            index.append(dict(path=f"ai/{gen}/clip{i}.wav", label=1, source=f"ai/{gen}"))
        index.append(dict(path=f"real/fma/{i:06d}.mp3", label=0, source="real/fma"))
    tr, va, te = _split_items(index)
    split_of = {}
    for name, part in (("tr", tr), ("va", va), ("te", te)):
        for it in part:
            stem = it["path"].rsplit("/", 1)[1].rsplit(".", 1)[0]
            group = stem.rsplit("_", 1)[0] if "suno" in stem else stem
            split_of.setdefault(group, set()).add(name)
    assert all(len(s) == 1 for s in split_of.values())
    assert len(te) > 0 and len(va) > 0


def test_mp3_roundtrip_preserves_length_and_removes_format_cue():
    from expressivemidi.detect.audio import SR, mp3_roundtrip
    rng = np.random.default_rng(0)
    y = (0.2 * rng.standard_normal(SR * 3)).astype(np.float32)
    z = mp3_roundtrip(y, 40)
    assert z.shape == y.shape and z.dtype == np.float32
    assert 0.1 < np.std(z) / np.std(y) < 1.5      # still the same signal level
    assert not np.allclose(z, y, atol=1e-3)       # the codec was actually applied


def test_train_refuses_cache_from_other_frontend(tmp_path):
    import json
    from expressivemidi.detect import audio
    (tmp_path / "index.json").write_text("[]")
    (tmp_path / "frontend.json").write_text(json.dumps(dict(frontend="sr32k-old")))
    with pytest.raises(SystemExit, match="front-end"):
        audio.train(str(tmp_path), str(tmp_path / "x.pt"), epochs=1)


def test_list_tracks_skips_macos_metadata(tmp_path):
    from expressivemidi.detect.audio import list_tracks
    for rel in ("ai/musicgen/a.wav", "ai/musicgen/._a.wav", "ai/__MACOSX/musicgen/._a.wav",
                "real/fma/000/000002.mp3", "real/fma/000/notes.txt"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    got = sorted((os.path.relpath(p, tmp_path).replace(os.sep, "/"), lab, src)
                 for p, lab, src in list_tracks(str(tmp_path)))
    assert got == [("ai/musicgen/a.wav", 1, "ai/musicgen"), ("real/fma/000/000002.mp3", 0, "real/fma")]


def test_audio_cache_keys_are_distinct():
    from expressivemidi.detect.audio import _cache_key
    keys = {_cache_key(f"data/audio/real/fma/{i:06d}.mp3") for i in range(100000)}
    assert len(keys) == 100000


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
