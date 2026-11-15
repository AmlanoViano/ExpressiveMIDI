# ExpressiveMIDI

Two everyday tools for people who work with music files:

1. **Humanise.** Make programmed or quantised MIDI sound like a person played it: timing, rubato, dynamics, articulation and sustain pedal, learned from real concert pianists playing from real scores.
2. **Detect.** Check whether music was made by a machine or an AI.
   - **MIDI:** was this played by a human, programmed in a sequencer, randomised by a "humanise" button, or rendered by an expressive-performance model?
   - **Audio:** is this track AI-generated (Suno, Udio, MusicGen, …)? Reliable only for the generators it was trained on; see the measured limits below.

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

Measured on 85 performances of 16 pieces that were never used in training or validation:

| Target | Model error | No-expression baseline | Correlation with real pianists |
|---|---|---|---|
| Micro-timing | 29.1 ms | 32.0 ms | 0.47 |
| Velocity | 11.8 (MIDI units) | 13.4 | 0.44 |
| Articulation (log length ratio) | 0.48 | 0.58 | 0.52 |
| Rubato (log beat-length ratio) | 0.154 | 0.134 | 0.23 |
| Sustain pedal down | 69.8% accuracy | 69.4% | n/a |

Micro-timing, dynamics and articulation are learned well. Two targets are weak, and the code works around them:
- **Rubato:** the correlation is positive, but the absolute error is worse than applying no rubato. Pianists disagree a lot on large-scale tempo. The humaniser therefore keeps only the phrase-level shape and removes slow drift, so a piece stays within a few hundred ms of the grid.
- **Pedal:** the model is no better than the majority-class baseline at predicting it from the score alone. It only decides *whether* a passage is pedalled. *Where* the pedal changes comes from a harmony rule that re-pedals when new notes would clash with what is ringing, as pianists do.

Training notes: about 8 minutes per epoch on a 4-core CPU. Validation loss is best after the first epoch. More epochs, even with transposition and noise augmentation, start memorising the roughly 200 training pieces. The shipped model is that best checkpoint.

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

Held-out results (pieces and drum sessions never seen in training; 1,454 files):

| | precision | recall |
|---|---|---|
| human | 0.96 | 0.95 |
| programmed | 0.95 | 0.94 |
| random jitter | 0.83 | 0.78 |
| model-humanised | 0.85 | 0.93 |

Overall accuracy is 92%, and the human-vs-machine ROC AUC is 0.992. On piano files accuracy is 99.1%; on drums it is 87.7%. As an extra check, 40 random MAESTRO performances that are not part of ASAP were all classified as human. ExpressiveMIDI's own output is detected as "model". That is intended: humanised files should not pass as real performances.

**Limits.** It judges how the notes were *produced or performed*, not who *composed* them. An AI-composed piece that a human played in looks human; a human composition typed into a sequencer looks programmed. Live-played MIDI from non-piano controllers is under-represented in training.

### Audio AI-music detector (included, trained)

This is a residual CNN on 6-second log-mel excerpts. Track scores are averaged over up to 30 excerpts. Training includes robustness augmentation (gain, extra band-limiting, resampling, EQ, noise, SpecAugment).

**Format-matched front-end.** Every track, real or AI, is resampled to **16 kHz** and passed through a **32–64 kbps MP3 round trip** before analysis. Inference uses the same front-end at 48 kbps. This is necessary because of how the public datasets are distributed:
- all SONICS (Suno/Udio) audio is 16 kHz mono MP3 at about 36 kbps;
- all FakeMusicCaps audio is 16 kHz WAV;
- FMA is 44.1 kHz MP3 at 128–320 kbps.

Without the shared front-end, "nothing above 8 kHz" or "low bit rate" alone would separate AI from real perfectly, and the detector would learn the file format instead of the music. The trade-off is that the model cannot use any detail above 8 kHz.

#### Results (shipped model, `models/audio_detector.pt`)

**Training data.** Real and AI sources differ, so expect some style mismatch.

| Side | Source | Tracks | Notes |
|---|---|---|---|
| real | FMA `fma_medium` | 24,980 | 30-s clips, 16 genres; 20 known-corrupt files skipped |
| ai | SONICS, Suno (chirp v2/v3/v3.5) | 4,965 | parts 1–2 of 10; full songs (2–4 min), all with vocals |
| ai | SONICS, Udio | 5,035 | parts 1–2 of 10; 30–130 s |
| ai | FakeMusicCaps: MusicGen, AudioLDM2, MusicLDM, Mustango, Stable Audio Open | 2,500 each | 10-s clips; the same 2,500 MusicCaps prompts for every generator |

That is 47,480 tracks in total: 24,980 real and 22,500 AI. They are split by *prompt group*, not by file, so the train/validation/test splits share no prompt groups:
- the two SONICS songs generated from one prompt always land in the same split;
- the same MusicCaps prompt rendered by different generators always lands in the same split.

The split is 37,923 train, 4,772 validation and 4,785 test tracks.

Training ran on an RTX 5060 Laptop GPU at about 25 min per epoch. It was **stopped after epoch 10 of a 20-epoch budget, at the user's request.** Validation AUC had been 1.000 (rounded) since epoch 2. The shipped checkpoint is the best by validation AUC (epoch 9, validation AUC 0.99998).

**Held-out test split** (generators seen in training, unseen tracks): accuracy **99.79%**, ROC AUC **0.99999** on 4,785 tracks.

| Test source | Tracks | Accuracy |
|---|---|---|
| real/fma | 2,490 | 99.92% (2 false alarms) |
| ai/suno | 474 | 100.0% |
| ai/udio | 496 | 100.0% |
| ai/musicldm | 265 | 100.0% |
| ai/audioldm2 | 265 | 99.62% |
| ai/mustango | 265 | 99.62% |
| ai/musicgen | 265 | 99.25% |
| ai/stable_audio | 265 | 98.49% |

**Real music from a different source.** 1,143 full-length tracks from MTG-Jamendo (tar parts 00–01 of `raw_30s/audio-low`) were never used in training. They were scored with the normal `detect` path: **1,142 classed human, 1 false alarm (0.09%)**, median P(AI) about 0.000002.

**Shortcut checks:**
- Accuracy is the same at every excerpt position within a track (99.5–99.7%). Dropping the first excerpt (song intros) leaves track accuracy at 99.75%.
- FMA false alarms are 0% in every genre except Experimental (1 of 224) and Jazz (1 of 38).
- SONICS songs all have vocals, but vocal Jamendo and FMA tracks are not flagged. So "has vocals" is not what the model learned.

**Unseen generators: the important limitation.** To test generalisation, a separate model was trained with one generator removed entirely, then tested on it. These hold-out models were trained for **4 epochs** to save time, fewer than the main model.

| Held out | Detected at the normal 0.5 threshold | AUC, held-out vs real test tracks | Detected at a threshold giving 1% false alarms |
|---|---|---|---|
| Udio (5,035 songs) | **0.0%** | 0.886 | 2.3% |
| Stable Audio Open | *pending, not yet run* | | |

With Udio unseen, the model catches **none** of the Udio songs, even though Suno songs, which are similar, were in training. Every generator it *was* trained on is still caught at 99–100%. In practice: **this detector recognises the generators it was trained on. It should not be expected to catch a new or unseen generator.** Udio versions newer than those in SONICS, and services not listed above, count as unseen.

To run the pending Stable Audio hold-out:

```
.\venv\Scripts\expressivemidi.exe train-audio-detector --holdout-source stable_audio --epochs 4 --out models/audio_holdout_stable_audio.pt
.\venv\Scripts\python.exe scripts\holdout_separation.py models\audio_holdout_stable_audio.pt
```

**Other limitations:**
- Real music comes only from FMA (independent, mostly amateur releases). Commercially mastered chart music was only tested indirectly (Jamendo is also independent music). False-alarm rates on heavily produced commercial releases are not measured.
- Only 2 of the 10 SONICS parts were used: about 20% of SONICS' Suno and Udio songs.
- Everything above 8 kHz is discarded by design, so codec-level artefacts in the top octave cannot be used.
- In-distribution accuracy near 100% does not mean the problem is solved. It means these generators leave consistent fingerprints *at the versions in these datasets*.

#### Training the audio detector yourself

1. Put audio in this layout. The sub-folder name becomes the "source" used in the reports:

   ```
   data/audio/real/<source>/...   e.g. real/fma, real/my_library, real/jamendo
   data/audio/ai/<source>/...     e.g. ai/suno, ai/udio, ai/musicgen, ai/stable_audio
   ```

   Sources used for the shipped model:
   - [SONICS](https://huggingface.co/datasets/awsaf49/sonics) on Hugging Face, CC BY-NC 4.0. Use the fake songs only; its "real" songs are YouTube links.
   - [FakeMusicCaps](https://zenodo.org/records/15063698) on Zenodo, CC BY-NC 4.0.
   - [FMA](https://github.com/mdeff/fma). Audio is under each artist's Creative Commons licence; the metadata is CC BY 4.0; for research use.
   - [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset), used for testing only. Per-track CC licences; non-commercial research use.

   Add tracks you generate yourself on the services you care about. Aim for a few thousand tracks per side, matched in genre if possible.

2. Prepare the spectrogram cache and train:

   ```bash
   expressivemidi audio-prepare --root data/audio --cache data/audio_cache --workers 6
   expressivemidi train-audio-detector --epochs 20 --require-gpu   # writes models/audio_detector.pt
   expressivemidi test-audio-detector                              # test-split metrics (e.g. after stopping early)
   ```

   Notes:
   - Training uses the GPU automatically with mixed precision. `--require-gpu` makes it fail instead of silently falling back to the CPU.
   - Batch size 32 uses about 5 GB of GPU memory.
   - The cache records the front-end version, and training refuses a cache built with a different one.

3. **Measure generalisation before trusting it.** Hold one generator out of training and test only on it:

   ```bash
   expressivemidi train-audio-detector --holdout-source udio --out models/audio_holdout_udio.pt
   python scripts/holdout_separation.py models/audio_holdout_udio.pt
   python scripts/eval_audio_detector.py --model models/audio_detector.pt \
       --fma-tracks data/raw/fma_metadata/tracks.csv --external real:data/audio_external/jamendo
   ```

The detector appears automatically in `expressivemidi detect` and in the web app.

**Limits.** No detector catches "any" AI music with certainty. As measured above, unseen generators are largely missed. New generator versions, heavy re-encoding, mastering or re-recording all reduce accuracy. Treat the output as evidence, never as proof, and never use it alone to accuse someone.

---

## Examples

`examples/` has before/after files you can open in any DAW or in the web app:
- `ravel_pavane_*` and `chopin_barcarolle_*`: opening sections of two pieces from the held-out test split, with rubato and pedal.
- `band_loop_*`: piano, bass and drums. Rubato stays off automatically, and the drums get a groove.

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
models/                  trained weights (humaniser.pt, midi_detector.joblib, audio_detector.pt)
scripts/                 audio-detector diagnostics (shortcut checks, held-out-generator separation)
tests/
experiments/legacy/      v1 checkpoints (kept for reference, not used)
```

## Licences and credits

Code: GPL-3.0 (see LICENSE). The **trained weights** are derived from ASAP and MAESTRO (CC BY-NC-SA 4.0) and the Groove MIDI Dataset (CC BY 4.0); the audio detector weights are trained on SONICS and FakeMusicCaps (both CC BY-NC 4.0) and FMA (per-artist CC licences, research use). All weights are for **non-commercial use**. Humanised MIDI you create from your own music is yours.

- Foscarin et al., *ASAP: a dataset of aligned scores and performances for piano transcription*, ISMIR 2020.
- Hawthorne et al., *Enabling factorized piano music modeling and generation with the MAESTRO dataset*, ICLR 2019.
- Gillick et al., *Learning to Groove with Inverse Sequence Transformations*, ICML 2019 (Groove MIDI Dataset).
