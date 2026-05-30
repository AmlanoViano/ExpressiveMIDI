# ExpressiveMIDI

Most MIDI files sound robotic because every note lands exactly on the beat, something that is not possible when playing physically. The micro-timing deviations that make a performance feel alive and genuine are completely absent.

ExpressiveMIDI is an attempt to learn those deviations from real performances and apply them to any MIDI file.

---

## Quick Start

If you just want to humanise a MIDI file:

```bash
git clone https://github.com/AmlanoViano/ExpressiveMIDI.git
cd ExpressiveMIDI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash humanise.sh
```

The script will ask for your input file, output name, and strength.

---

## How it works

The model is trained on [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) - 962 professional piano competition recordings where every keypress is captured with millisecond precision. For each note, we compute how much the pianist deviated from a quantised grid (the "score"), and train the model to predict those deviations from musical context alone: pitch, rhythm, velocity, phrase position, harmonic interval, and local tempo.

At inference time, the model reads any MIDI file and outputs a per-note timing deviation in milliseconds. These are applied back to the file, shifting notes slightly earlier or later in a musically coherent way.

---

## Architecture

After testing a BiLSTM baseline and a pure Transformer, the best results came from a hybrid:

- **CNN layers** (kernels 3, 5, 7) capture local note-group patterns - phrase shapes, ornaments, chord approaches
- **Transformer encoder** (4 heads, 3 layers) models long-range dependencies - how the timing of a note 30 notes ago influences the current one
- **Separate prediction head** with GELU activations

| Model | MAE (ms) |
|---|---|
| BiLSTM baseline | 8.06 |
| BiLSTM + richer features | 7.49 |
| CNN + Transformer hybrid | **7.33** |

Training used mixed precision (AMP) on an RTX 2060 with early stopping, AdamW, and ReduceLROnPlateau scheduling. Each run takes roughly 20 minutes on the full 5.6M note dataset.

---

## Features

Each note is represented by 21 features:

- **Pitch:** absolute, pitch class, octave, normalised
- **Intervals:** to previous and next note, absolute interval, melodic contour
- **Timing:** local IOI, tempo reference, IOI ratio, continuous beat position, beat strength
- **Duration:** normalised, ratio to local tempo
- **Velocity:** normalised, deviation from local mean
- **Chord context:** whether the note is part of a chord, local chord density
- **Phrase:** position within phrase, phrase start/end flags

---

## Setup

```bash
git clone https://github.com/AmlanoViano/ExpressiveMIDI.git
cd ExpressiveMIDI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download MAESTRO v3 (MIDI only, ~50MB):

```bash
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip -d data/raw/
```

---

## Training

Process the dataset:

```bash
python -c "
from src.data.loader import process_dataset
process_dataset('data/raw/maestro-v3.0.0', 'data/processed', split='train')
process_dataset('data/raw/maestro-v3.0.0', 'data/processed', split='validation')
"
```

Train the hybrid model:

```bash
python -m src.training.train --model hybrid --lr 3e-4 --epochs 60 --patience 8
```

Other options: `--model lstm`, `--model transformer`

---

## Humanise any MIDI

**Using the launcher (easiest):**

```bash
bash humanise.sh
```

**Using the CLI directly:**

```bash
python -m src.inference.humanise \
  --input your_file.mid \
  --output humanised.mid \
  --model experiments/best_hybrid_*.pt \
  --strength 1.0
```

The `--strength` parameter scales the predicted deviations. `0.5` is subtle, `1.0` is natural, `2.0` is exaggerated.

---

## Project Structure

```
ExpressiveMIDI/
├── src/
│   ├── data/
│   │   ├── loader.py       # MAESTRO parsing and deviation extraction
│   │   ├── features.py     # 21 note-level features
│   │   └── dataset.py      # PyTorch Dataset with sliding window
│   ├── models/
│   │   ├── baseline.py     # Bidirectional LSTM
│   │   ├── transformer.py  # Transformer encoder
│   │   └── hybrid.py       # CNN + Transformer (best performing)
│   ├── training/
│   │   └── train.py        # Training loop with early stopping and AMP
│   └── inference/
│       └── humanise.py     # Apply model to any MIDI file
├── humanise.sh             # One-click launcher
├── experiments/            # Model checkpoints and training histories
├── notebooks/              # Exploratory analysis
└── docs/                   # Writeup and figures
```
---

## Acknowledgements

Built on [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) (Hawthorne et al., 2019) and trained using PyTorch with the [pretty_midi](https://github.com/craffel/pretty-midi) library for MIDI parsing.
