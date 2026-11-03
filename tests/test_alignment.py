import numpy as np

from expressivemidi.data.asap import match_notes


def test_match_notes_with_rubato_and_ornaments():
    rng = np.random.default_rng(0)
    s_on = np.arange(40) * 0.5
    s_pitch = 60 + (np.arange(40) % 4)
    pred = s_on * 1.1                       # where beat alignment says each note should be
    p_on = pred + rng.normal(0, 0.02, 40)   # performed with small deviations
    p_pitch = s_pitch.copy()
    # an extra ornament note that has no score counterpart
    p_on = np.append(p_on, 3.0); p_pitch = np.append(p_pitch, 61)
    order = np.argsort(p_on); p_on, p_pitch = p_on[order], p_pitch[order]
    m = match_notes(s_pitch, pred, p_on, p_pitch, np.full(40, 0.2))
    assert (m >= 0).all()
    assert np.allclose(p_on[m], pred, atol=0.1)
    assert len(set(m)) == 40                # one-to-one
