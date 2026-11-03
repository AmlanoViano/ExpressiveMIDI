import numpy as np
import pretty_midi

from expressivemidi.humanise import TAG, Settings, humanise_file, humanise_midi, quantisation_ratio
from conftest import make_song


def _notes(inst):
    return sorted(inst.notes, key=lambda n: (n.start, n.pitch))


def test_preserves_structure(song, fake_model):
    model, ck, _ = fake_model
    out, rep = humanise_midi(song, model, ck, Settings())
    assert len(out.instruments) == len(song.instruments)
    for a, b in zip(song.instruments, out.instruments):
        assert (a.program, a.is_drum, a.name, len(a.notes)) == (b.program, b.is_drum, b.name, len(b.notes))
    assert np.allclose(out.get_tempo_changes()[1], song.get_tempo_changes()[1])
    assert [(t.numerator, t.denominator) for t in out.time_signature_changes] == [(4, 4)]
    assert rep.notes == sum(len(i.notes) for i in song.instruments if not i.is_drum)
    assert rep.rubato == 0.0                      # drums present -> rubato auto-off
    assert any(TAG in t.text for t in out.text_events)


def test_changes_timing_and_velocity_but_stays_close(song, fake_model):
    model, ck, _ = fake_model
    out, rep = humanise_midi(song, model, ck, Settings())
    a, b = _notes(song.instruments[0]), _notes(out.instruments[0])
    shifts = np.array([abs(x.start - y.start) for x, y in zip(a, b)])
    assert shifts.max() > 0 and shifts.max() < 0.08
    assert len({n.velocity for n in b}) > 1
    assert all(1 <= n.velocity <= 127 for i in out.instruments for n in i.notes)
    assert all(n.start >= 0 and n.end > n.start for i in out.instruments for n in i.notes)


def test_no_same_pitch_overlap(song, fake_model):
    model, ck, _ = fake_model
    out, _ = humanise_midi(song, model, ck, Settings(strength=2.0, articulation=2.0))
    for inst in out.instruments:
        by = {}
        for n in inst.notes:
            by.setdefault(n.pitch, []).append(n)
        for lst in by.values():
            lst.sort(key=lambda n: n.start)
            assert all(x.end <= y.start + 1e-9 for x, y in zip(lst, lst[1:]))


def test_strength_zero_is_identity_except_tag(song, fake_model):
    model, ck, _ = fake_model
    out, rep = humanise_midi(song, model, ck, Settings(strength=0.0, drums=0.0, pedal="off"))
    for a, b in zip(song.instruments, out.instruments):
        for x, y in zip(_notes(a), _notes(b)):
            assert abs(x.start - y.start) < 1e-9 and x.velocity == y.velocity


def test_rubato_solo_piano_keeps_length_and_order(fake_model):
    model, ck, _ = fake_model
    song = make_song(drums=False)
    out, rep = humanise_midi(song, model, ck, Settings(rubato=1.0, timing=0.0))
    assert rep.rubato == 1.0
    assert abs(out.get_end_time() - song.get_end_time()) < 0.5
    # melody order preserved
    mel_in = [n.pitch for n in _notes(song.instruments[0]) if n.pitch >= 72]
    mel_out = [n.pitch for n in _notes(out.instruments[0]) if n.pitch >= 72]
    assert mel_in == mel_out


def test_pedal_generated_for_piano_only(fake_model):
    model, ck, _ = fake_model
    ck = dict(ck); ck["y_mean"] = list(ck["y_mean"])
    song = make_song(drums=False)
    out, rep = humanise_midi(song, model, ck, Settings(pedal="on"))
    cc = [c for c in out.instruments[0].control_changes if c.number == 64]
    assert rep.pedal_events == len(cc)
    assert all(c.value in (0, 127) for c in cc)
    assert not any(c.number == 64 for c in out.instruments[1].control_changes)


def test_quantisation_ratio(song):
    assert quantisation_ratio(song) == 1.0
    rng = np.random.default_rng(0)
    for inst in song.instruments:
        for n in inst.notes:
            n.start += rng.uniform(0.01, 0.03)
    assert quantisation_ratio(song) < 0.2


def test_file_roundtrip(song, fake_model, tmp_path):
    model, ck, path = fake_model
    src = tmp_path / "in.mid"; dst = tmp_path / "out.mid"
    song.write(str(src))
    rep = humanise_file(str(src), str(dst), path)
    back = pretty_midi.PrettyMIDI(str(dst))
    assert len(back.instruments) == 3 and back.instruments[2].is_drum
    assert rep.notes > 0
