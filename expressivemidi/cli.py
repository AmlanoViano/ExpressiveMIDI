"""Command-line interface:  expressivemidi <command> ...   (or python -m expressivemidi)"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HUMANISER = os.path.join(HERE, "models", "humaniser.pt")
DEFAULT_MIDI_DET = os.path.join(HERE, "models", "midi_detector.joblib")
DEFAULT_AUDIO_DET = os.path.join(HERE, "models", "audio_detector.pt")
MIDI_EXT = (".mid", ".midi")


def _collect(paths, exts):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for d, _, files in os.walk(p):
                out += [os.path.join(d, f) for f in sorted(files) if f.lower().endswith(exts)]
        else:
            out.append(p)
    return out


def cmd_humanise(a):
    from .humanise import Settings, humanise_file, load_model
    files = _collect(a.inputs, MIDI_EXT)
    if not files:
        sys.exit("No MIDI files found.")
    if a.output and len(files) > 1:
        sys.exit("--output only works with a single input; use --out-dir for several files.")
    model, ck = load_model(a.model)
    s = Settings(strength=a.strength, timing=a.timing, rubato=a.rubato, velocity=a.velocity,
                 articulation=a.articulation, pedal=a.pedal, drums=a.drums, seed=a.seed, tag=not a.no_tag)
    for f in files:
        base, _ = os.path.splitext(os.path.basename(f))
        out = a.output or os.path.join(a.out_dir or os.path.dirname(f) or ".", f"{base}_humanised.mid")
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
        rep = humanise_file(f, out, settings=s, model=model, ck=ck)
        print(f"{f} -> {out}")
        print(f"  {rep.notes} notes in {', '.join(rep.parts) or 'no pitched parts'}; drums: {rep.drum_notes}; "
              f"rubato {rep.rubato:.2f}; avg shift {rep.mean_abs_shift_ms:.1f} ms; "
              f"avg velocity change {rep.velocity_change:.1f}; pedal events {rep.pedal_events}")
        for w in rep.warnings:
            print(f"  ! {w}")


def cmd_detect(a):
    files = _collect(a.inputs, MIDI_EXT + (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"))
    results = []
    for f in files:
        if f.lower().endswith(MIDI_EXT):
            from .detect.midi_detector import detect_midi
            r = detect_midi(f, a.midi_model)
        else:
            if not os.path.exists(a.audio_model):
                r = dict(file=os.path.basename(f), type="audio", error=(
                    "No audio detector installed. Train one with `expressivemidi audio-prepare` + "
                    "`expressivemidi train-audio-detector` (see README)."))
            else:
                from .detect.audio import detect_audio
                r = detect_audio(f, a.audio_model)
        results.append(r)
        if not a.json:
            _print_detection(r)
    if a.json:
        print(json.dumps(results, indent=2))


def _print_detection(r):
    print(f"\n{r['file']}  [{r['type']}]")
    if "error" in r:
        print(f"  {r['error']}")
        return
    if r["type"] == "midi":
        print(f"  verdict: {r['verdict_text']}")
        for c, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"    {c:<11} {p:6.1%}  {'#' * int(p * 30)}")
        print("  evidence:")
        for e in r["evidence"]:
            print(f"    - {e}")
    else:
        print(f"  verdict: {r['verdict_text']}  (P(AI) = {r['p_ai']:.1%}, "
              f"{r['excerpts_flagged']:.0%} of {r['excerpts']} excerpts flagged)")
    for n in r.get("notes", []):
        print(f"  note: {n}")


def cmd_build_dataset(a):
    from .data.asap import build_dataset
    build_dataset(a.asap, a.out, workers=a.workers)


def cmd_train_humaniser(a):
    from .train_humaniser import train
    train(data_dir=a.data, out_path=a.out, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
          patience=a.patience, max_minutes=a.max_minutes)


def cmd_test_humaniser(a):
    from .train_humaniser import test
    m = test(a.model, a.data)
    for k in ("tempo", "micro", "vel", "artic"):
        print(f"{k:>6}: MAE {m[k+'_mae']:.4f} (no-expression baseline {m[k+'_baseline_mae']:.4f}), "
              f"corr {m[k+'_corr']:.3f}")
    for k in ("pedal", "repedal"):
        print(f"{k:>6}: acc {m[k+'_acc']:.3f} (majority baseline {m[k+'_baseline_acc']:.3f})")


def cmd_train_midi_detector(a):
    from .detect.midi_detector import build_training_set, train
    if not a.skip_build:
        build_training_set(a.asap, a.groove, a.humaniser, a.data, workers=a.workers)
    train(a.data, a.out)


def cmd_audio_prepare(a):
    from .detect.audio import prepare
    prepare(a.root, a.cache, n_excerpts=a.excerpts, workers=a.workers, augment=not a.no_augment)


def cmd_train_audio(a):
    from .detect.audio import train
    train(a.cache, a.out, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
          holdout_source=a.holdout_source, workers=a.workers)


def cmd_serve(a):
    from webapp.app import create_app
    create_app().run(host=a.host, port=a.port, debug=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="expressivemidi", description="Humanise MIDI and detect machine/AI-made music.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("humanise", aliases=["humanize"], help="humanise MIDI files or folders")
    h.add_argument("inputs", nargs="+")
    h.add_argument("-o", "--output")
    h.add_argument("--out-dir")
    h.add_argument("--model", default=DEFAULT_HUMANISER)
    h.add_argument("--strength", type=float, default=1.0, help="overall amount (0.5 subtle, 1 natural, 1.5 bold)")
    h.add_argument("--timing", type=float, default=1.0, help="micro-timing amount")
    h.add_argument("--rubato", type=float, default=None,
                   help="tempo flexibility; default: on for solo/no-drum files, off when drums are present")
    h.add_argument("--velocity", type=float, default=1.0)
    h.add_argument("--articulation", type=float, default=1.0)
    h.add_argument("--pedal", choices=["auto", "on", "off"], default="auto")
    h.add_argument("--drums", type=float, default=1.0, help="drum groove humanisation amount (0 = untouched)")
    h.add_argument("--seed", type=int, default=0, help="change for a different drum take")
    h.add_argument("--no-tag", action="store_true", help="don't mark the file as humanised")
    h.set_defaults(fn=cmd_humanise)

    d = sub.add_parser("detect", help="check MIDI or audio files for machine/AI generation")
    d.add_argument("inputs", nargs="+")
    d.add_argument("--json", action="store_true")
    d.add_argument("--midi-model", default=DEFAULT_MIDI_DET)
    d.add_argument("--audio-model", default=DEFAULT_AUDIO_DET)
    d.set_defaults(fn=cmd_detect)

    s = sub.add_parser("serve", help="start the local web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5050)
    s.set_defaults(fn=cmd_serve)

    b = sub.add_parser("build-dataset", help="build humaniser training data from ASAP")
    b.add_argument("--asap", default="data/raw/asap-dataset")
    b.add_argument("--out", default="data/processed")
    b.add_argument("--workers", type=int, default=4)
    b.set_defaults(fn=cmd_build_dataset)

    t = sub.add_parser("train-humaniser")
    t.add_argument("--data", default="data/processed")
    t.add_argument("--out", default=DEFAULT_HUMANISER)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--max-minutes", type=float, default=None)
    t.set_defaults(fn=cmd_train_humaniser)

    th = sub.add_parser("test-humaniser", help="held-out test metrics")
    th.add_argument("--model", default=DEFAULT_HUMANISER)
    th.add_argument("--data", default="data/processed")
    th.set_defaults(fn=cmd_test_humaniser)

    m = sub.add_parser("train-midi-detector")
    m.add_argument("--asap", default="data/raw/asap-dataset")
    m.add_argument("--groove", default="data/raw/groove")
    m.add_argument("--humaniser", default=DEFAULT_HUMANISER)
    m.add_argument("--data", default="data/processed/midi_detector.npz")
    m.add_argument("--out", default=DEFAULT_MIDI_DET)
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--skip-build", action="store_true")
    m.set_defaults(fn=cmd_train_midi_detector)

    ap = sub.add_parser("audio-prepare", help="cache spectrogram excerpts from data/audio/{real,ai}")
    ap.add_argument("--root", default="data/audio")
    ap.add_argument("--cache", default="data/audio_cache")
    ap.add_argument("--excerpts", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-augment", action="store_true")
    ap.set_defaults(fn=cmd_audio_prepare)

    ta = sub.add_parser("train-audio-detector")
    ta.add_argument("--cache", default="data/audio_cache")
    ta.add_argument("--out", default=DEFAULT_AUDIO_DET)
    ta.add_argument("--epochs", type=int, default=20)
    ta.add_argument("--batch-size", type=int, default=32)
    ta.add_argument("--lr", type=float, default=1e-3)
    ta.add_argument("--workers", type=int, default=4)
    ta.add_argument("--holdout-source", default=None,
                    help="e.g. 'udio': exclude ai/udio from training and test on it (unseen-generator check)")
    ta.set_defaults(fn=cmd_train_audio)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
