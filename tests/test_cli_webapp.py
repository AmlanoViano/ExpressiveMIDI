import io

import pretty_midi

from expressivemidi.cli import main
from conftest import make_song


def test_cli_humanise_folder(fake_model, tmp_path, capsys):
    _, _, model_path = fake_model
    src = tmp_path / "in"; src.mkdir()
    for k in range(2):
        make_song().write(str(src / f"s{k}.mid"))
    main(["humanise", str(src), "--out-dir", str(tmp_path / "out"), "--model", model_path])
    outs = sorted((tmp_path / "out").iterdir())
    assert [p.name for p in outs] == ["s0_humanised.mid", "s1_humanised.mid"]
    assert "notes" in capsys.readouterr().out


def test_webapp_humanise(fake_model, monkeypatch):
    import webapp.app as wa
    _, _, model_path = fake_model
    monkeypatch.setattr(wa, "DEFAULT_HUMANISER", model_path)
    client = wa.create_app().test_client()
    buf = io.BytesIO(); make_song().write(buf); buf.seek(0)
    r = client.post("/api/humanise", data={"file": (buf, "song.mid"), "strength": "1.2", "rubato": "auto"},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.json
    assert r.json["filename"] == "song_humanised.mid" and r.json["report"]["notes"] > 0
    bad = client.post("/api/humanise", data={"file": (io.BytesIO(b"x"), "a.txt")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400
    assert client.get("/").status_code == 200
