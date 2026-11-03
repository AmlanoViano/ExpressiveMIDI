# ExpressiveMIDI

Two everyday tools for people who work with music files:

1. **Humanise.** Make programmed or quantised MIDI sound like a person played it: timing, rubato, dynamics, articulation and sustain pedal, learned from real concert pianists playing from real scores.
2. **Detect.** Check whether music was made by a machine or an AI.
   - **MIDI:** was this played by a human, programmed in a sequencer, randomised by a "humanise" button, or rendered by an expressive-performance model?
   - **Audio:** is this track AI-generated (Suno, Udio, MusicGen, …)? You train this one on your own GPU; see below.

Everything runs locally. Nothing is uploaded anywhere.

---

## Quick start

```bash
git clone https://github.com/AmlanoViano/ExpressiveMIDI.git
cd ExpressiveMIDI
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

**Web app** (drag and drop, before/after playback in the browser):

```bash
expressivemidi serve
```

Then open <http://127.0.0.1:5050>.

**Command line:**

```bash
# one file
expressivemidi humanise song.mid                      # -> song_humanised.mid
expressivemidi humanise song.mid -o take2.mid --strength 1.3 --seed 2

# a whole folder
expressivemidi humanise exports/ --out-dir humanised/

# check files (MIDI and audio can be mixed; folders work too)
expressivemidi detect suspicious.mid track.mp3
expressivemidi detect inbox/ --json > report.json
```

### Humanise options

| Option | Default | What it does |
|---|---|---|
| `--strength` | 1.0 | Overall amount. 0.5 is subtle, 1 is natural, 1.5 is bold. |
| `--timing` | 1.0 | Micro-timing: chord spread, melody lead, push and pull. |
| `--rubato` | auto | Tempo breathing. **Auto** turns it on for files without drums and off when drums are present, so songs with a beat stay on the grid. It is phrase-level only, so your chosen tempo is kept. |
| `--velocity` | 1.0 | Dynamics and accents. |
| `--articulation` | 1.0 | Note lengths (legato/staccato shaping). |
| `--pedal` | auto | Sustain pedal for piano parts. `auto` adds pedal only if the part has none. `on` replaces existing pedal. `off` never adds it. |
| `--drums` | 1.0 | Groove humanisation for drum tracks (0 leaves them untouched). |
| `--seed` | 0 | Different drum take. |
| `--no-tag` | | Don't write the small "Humanised with ExpressiveMIDI" text marker. |

What is kept: every track, instrument and program, drum tracks, the tempo map, time and key signatures, existing controllers and pitch bends. Only note timing, length and velocity change, plus pedal if requested. If the input already looks performed (not on a grid), you get a warning, because humanising twice stacks the deviations.

---

## What changed in v2 (and why v1 didn't work)

v1 trained on MAESTRO, which contains **performances only, with no scores**. To get a "deviation", it snapped each performed note to a fixed grid of about 38 ms built from the performance itself, and treated the remainder as the target. That remainder is noise:

- it is spread evenly across ±19 ms, which is exactly what chance would give;
- adding **10 ms of silence** to the start of a recording changes the targets completely (correlation −0.18 across 30 validation pieces);
- always predicting **zero** gives 9.6 ms error. The best v1 model was reported at 7.33 ms, but the v1 notebook measured it at 19.2 ms on validation (strings: 42.7 ms).

**MAESTRO itself is an excellent dataset. It was just used without the information needed to define expression.** Other v1 problems:
- The output merged everything into one piano track and dropped drums.
- Feature scaling was recomputed per file instead of saved with the model.
- Pedal values left the sustain half-down all the time.
- "Phrase" features were just note-index mod 16.
- Training crashed on CPU.
- `requirements.txt` pinned nightly CUDA builds.
- The strings model's "vibrato" target was the last pitch-bend value before each note.

v2 trains on **[ASAP](https://github.com/fosfrancesco/asap-dataset)**: 1,067 MAESTRO performances aligned with their **scores** and with beat annotations for both. For every notated note, the matching performed note is found by beat alignment. This matches 94.7% of notes and keeps 1,027 performances. From each match we measure:

| Target | Meaning |
|---|---|
| tempo | Local beat length versus the piece's median (rubato). |
| micro | Performed onset minus the beat-interpolated onset, in ms (asynchrony, melody lead). |
| vel | Velocity relative to the piece average. |
| artic | log(performed length ÷ notated length at the local tempo). |
| pedal / repedal | Sustain down after the onset; pedal changed at the onset. |

The inputs are 22 features that exist in any quantised score: pitch, beat-relative durations, metre position, chord structure, rests and density. Training and real use therefore see the same kind of data. The model is a CNN + Transformer (0.9M parameters) and runs fast on a CPU. Pieces are split by *piece*, never by performance, so the test pieces are completely unseen.

### Humaniser results (held-out test pieces)

<!-- RESULTS:HUMANISER -->

"Baseline" means applying no expression, which is what a quantised file does. Lower error and a positive correlation mean the model predicts what real pianists did with music it has never seen. Expressive performance is partly a personal choice, so the correlations will never approach 1.

The strings model was removed. Use the general model for melodic instruments (articulation is automatically reduced to half for non-piano parts) and `--rubato 0` for ensemble material.

---

## AI and machine detection

### MIDI provenance detector (included, trained)

This is a gradient-boosted classifier over 30 file-level features. It deliberately ignores file-format clues such as track count, tick resolution and meta events, which are trivial to fake. Instead it measures musical evidence:
- how exactly notes sit on the grid;
- whether timing deviations drift smoothly (human) or jump independently (random jitter);
- the shape of chord asynchrony and melody lead;
- velocity structure and chord voicing;
- whether note lengths are exact grid values.

Training data comes from the same pieces in every class:
- **human:** ASAP/MAESTRO piano performances and Groove MIDI Dataset drum performances;
- **programmed:** the scores and quantised grooves, with constant, stepped or accented velocities and fixed swing;
- **random:** programmed plus DAW-style independent jitter;
- **model:** rendered by this project's humaniser.

<!-- RESULTS:MIDIDETECTOR -->

**Limits.** It judges how the notes were *produced or performed*, not who *composed* them. An AI-composed piece that a human played in looks human; a human composition typed into a sequencer looks programmed. Live-played MIDI from non-piano controllers is under-represented in training.

### Audio AI-music detector (pipeline included, you train it)

This is a residual CNN on 6-second log-mel excerpts at 32 kHz. Track scores are averaged over up to 30 excerpts. Training includes robustness augmentation: band-limiting like low-bitrate MP3, resampling, EQ, noise and SpecAugment. No weights are shipped, because the datasets have to be downloaded and training needs a GPU.

#### Training the audio detector

1. Put audio in this layout. The sub-folder name becomes the "source" used in the reports:

   ```
   data/audio/real/<source>/...   e.g. real/fma, real/my_library, real/jamendo
   data/audio/ai/<source>/...     e.g. ai/suno, ai/udio, ai/musicgen, ai/stable_audio
   ```

   Good sources:
   - **AI:** the SONICS dataset (Suno and Udio songs, on Hugging Face), FakeMusicCaps (MusicGen, AudioLDM2, Stable Audio and others, on Zenodo), plus tracks you generate yourself on the services you care about.
   - **Real:** FMA (Free Music Archive), MTG-Jamendo, or your own library.
   - Aim for at least a few thousand tracks per side, matched in genre if possible. Otherwise the model learns "genre" instead of "AI".

2. Prepare the spectrogram cache and train:

   ```bash
   expressivemidi audio-prepare --root data/audio --cache data/audio_cache --workers 8
   expressivemidi train-audio-detector --epochs 20      # writes models/audio_detector.pt
   ```

   It uses the GPU automatically (mixed precision). An RTX 2060 handles batch size 32.

3. **Measure generalisation before trusting it.** Hold one generator out of training and test only on it:

   ```bash
   expressivemidi train-audio-detector --holdout-source udio --out models/audio_holdout_udio.pt
   ```

   If accuracy on the held-out generator is much lower than on the others, which is common, the detector won't reliably catch new or unseen generators. Add more generator variety.

The detector then appears automatically in `expressivemidi detect` and in the web app.

**Limits.** No detector catches "any" AI music with certainty. New generators, heavy re-encoding, mastering or re-recording all reduce accuracy. Treat the output as evidence, never as proof, and never use it alone to accuse someone.

---

## Retraining everything yourself

```bash
pip install -e '.[dev]'
mkdir -p data/raw
git clone --depth 1 https://github.com/fosfrancesco/asap-dataset data/raw/asap-dataset
curl -LO https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
unzip groove-v1.0.0-midionly.zip -d data/raw/                  # -> data/raw/groove

expressivemidi build-dataset                     # ASAP -> data/processed/asap_*.npz
expressivemidi train-humaniser --epochs 30       # GPU or CPU (CPU: ~8 min/epoch)
expressivemidi test-humaniser                    # held-out metrics
expressivemidi train-midi-detector               # uses the humaniser for the "model" class
pytest                                           # test suite
```

## Project structure

```
expressivemidi/
  features.py            score features + beat grid (shared by training and inference)
  data/asap.py           ASAP score<->performance alignment and targets
  models/humaniser_net.py
  train_humaniser.py
  humanise.py            inference: multi-track, rubato warp, pedal, drums
  detect/midi_features.py, midi_detector.py
  detect/audio.py        audio AI detector (prepare / train / detect)
  cli.py
webapp/                  Flask app (expressivemidi serve)
models/                  trained weights (humaniser.pt, midi_detector.joblib)
tests/
experiments/legacy/      v1 checkpoints (kept for reference, not used)
```

## Licences and credits

Code: GPL-3.0 (see LICENSE). The **trained weights** are derived from ASAP and MAESTRO (CC BY-NC-SA 4.0) and the Groove MIDI Dataset (CC BY 4.0), so the weights are for **non-commercial use**. Humanised MIDI you create from your own music is yours.

- Foscarin et al., *ASAP: a dataset of aligned scores and performances for piano transcription*, ISMIR 2020.
- Hawthorne et al., *Enabling factorized piano music modeling and generation with the MAESTRO dataset*, ICLR 2019.
- Gillick et al., *Learning to Groove with Inverse Sequence Transformations*, ICML 2019 (Groove MIDI Dataset).
