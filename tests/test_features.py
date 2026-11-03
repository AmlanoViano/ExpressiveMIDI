import numpy as np

from expressivemidi.features import FEATURE_NAMES, BeatGrid, score_features


def test_beatgrid_roundtrip_and_extrapolation():
    g = BeatGrid(np.arange(0, 10, 0.5))
    t = np.array([-1.0, 0.25, 3.3, 12.0])
    assert np.allclose(g.to_seconds(g.to_beats(t)), t)
    assert np.isclose(g.to_beats([12.0])[0], 24.0)


def test_features_shape_and_chords(song):
    notes = sorted((n for n in song.instruments[0].notes), key=lambda n: (n.start, n.pitch))
    on = np.array([n.start for n in notes]); off = np.array([n.end for n in notes])
    pitch = np.array([n.pitch for n in notes])
    g = BeatGrid(song.get_beats(), song.get_downbeats())
    X = score_features(on, off, pitch, g)
    assert X.shape == (len(notes), len(FEATURE_NAMES))
    assert np.isfinite(X).all()
    f = dict(zip(FEATURE_NAMES, X[0]))
    assert f["metric_strength"] == 1.0          # first note is a downbeat
    assert f["chord_size"] > 0.25               # part of a 4-note chord (triad + melody)
    assert X[:, FEATURE_NAMES.index("chord_top")].sum() > 0


def test_features_are_tempo_aware_but_grid_relative():
    """Same rhythm at different tempi -> identical beat-relative features."""
    def feats(tempo):
        beat = 60 / tempo
        on = np.arange(16) * beat / 2; off = on + beat / 2; p = 60 + np.arange(16) % 5
        return score_features(on, off, p, BeatGrid(np.arange(0, 20) * beat))
    a, b = feats(90), feats(140)
    i = FEATURE_NAMES.index("log_dur_secs")
    assert np.allclose(np.delete(a, i, 1), np.delete(b, i, 1), atol=1e-5)
