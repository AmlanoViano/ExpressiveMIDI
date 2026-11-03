"""Local web app: humanise MIDI files and check files for machine/AI generation.

    expressivemidi serve            # then open http://127.0.0.1:5050
"""
from __future__ import annotations

import base64
import os
import shutil
import tempfile
import threading

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from expressivemidi import __version__
from expressivemidi.cli import DEFAULT_AUDIO_DET, DEFAULT_HUMANISER, DEFAULT_MIDI_DET
from expressivemidi.humanise import Settings, humanise_file, load_model

MIDI_EXT = (".mid", ".midi")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")


def _float(form, key, default, lo=0.0, hi=3.0):
    try:
        return min(max(float(form.get(key, default)), lo), hi)
    except (TypeError, ValueError):
        return default


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    lock = threading.Lock()
    state = {}

    def humaniser():
        with lock:
            if "model" not in state:
                state["model"] = load_model(DEFAULT_HUMANISER)
        return state["model"]

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__,
                               audio_ready=os.path.exists(DEFAULT_AUDIO_DET),
                               midi_ready=os.path.exists(DEFAULT_MIDI_DET))

    @app.post("/api/humanise")
    def api_humanise():
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(MIDI_EXT):
            return jsonify(error="Please upload a .mid / .midi file."), 400
        form = request.form
        rubato = form.get("rubato", "auto")
        s = Settings(
            strength=_float(form, "strength", 1.0),
            timing=_float(form, "timing", 1.0),
            velocity=_float(form, "velocity", 1.0),
            articulation=_float(form, "articulation", 1.0),
            rubato=None if rubato == "auto" else _float(form, "rubato", 1.0),
            pedal=form.get("pedal", "auto") if form.get("pedal") in ("auto", "on", "off") else "auto",
            drums=_float(form, "drums", 1.0),
            seed=int(_float(form, "seed", 0, 0, 1e6)),
        )
        tmp = tempfile.mkdtemp(prefix="emidi_")
        try:
            name = secure_filename(f.filename) or "input.mid"
            src = os.path.join(tmp, name)
            dst = os.path.join(tmp, "out.mid")
            f.save(src)
            model, ck = humaniser()
            rep = humanise_file(src, dst, settings=s, model=model, ck=ck)
            with open(dst, "rb") as fh:
                data = base64.b64encode(fh.read()).decode()
            base = os.path.splitext(name)[0]
            return jsonify(filename=f"{base}_humanised.mid", midi=data, report=rep.__dict__)
        except Exception as e:  # noqa: BLE001
            return jsonify(error=f"Could not process this file: {e}"), 500
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.post("/api/detect")
    def api_detect():
        files = request.files.getlist("files")
        if not files:
            return jsonify(error="No files uploaded."), 400
        results = []
        tmp = tempfile.mkdtemp(prefix="emidi_")
        try:
            for f in files:
                name = secure_filename(f.filename) or "upload"
                path = os.path.join(tmp, name)
                f.save(path)
                low = name.lower()
                try:
                    if low.endswith(MIDI_EXT):
                        from expressivemidi.detect.midi_detector import detect_midi
                        results.append(detect_midi(path, DEFAULT_MIDI_DET))
                    elif low.endswith(AUDIO_EXT):
                        if not os.path.exists(DEFAULT_AUDIO_DET):
                            results.append(dict(file=name, type="audio", error=(
                                "No audio detector model installed yet - see README "
                                "'Training the audio detector'.")))
                        else:
                            from expressivemidi.detect.audio import detect_audio
                            results.append(detect_audio(path, DEFAULT_AUDIO_DET))
                    else:
                        results.append(dict(file=name, error="Unsupported file type."))
                except Exception as e:  # noqa: BLE001
                    results.append(dict(file=name, error=f"Could not analyse: {e}"))
            return jsonify(results=results)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return app


if __name__ == "__main__":
    create_app().run(port=5050)
