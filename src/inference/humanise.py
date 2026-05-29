"""
humanise.py - Apply trained model to any MIDI file to add expressive timing.
Usage: python -m src.inference.humanise --input song.mid --output song_expressive.mid --model experiments/best_hybrid_*.pt
"""
import argparse
import torch
import numpy as np
import pretty_midi
import pandas as pd

from src.data.features import add_features, FEATURE_COLS
from src.models.hybrid import HybridTimingModel


def load_model(model_path: str, input_dim: int, device: torch.device):
    model = HybridTimingModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def midi_to_df(midi_path: str) -> tuple:
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                notes.append({
                    'pitch':          n.pitch,
                    'onset':          n.start,
                    'onset_score':    n.start,
                    'duration':       n.end - n.start,
                    'duration_score': n.end - n.start,
                    'velocity':       n.velocity,
                    'velocity_score': n.velocity,
                    'offset':         n.end,
                })
    df = pd.DataFrame(notes).sort_values('onset').reset_index(drop=True)
    return midi, df


def predict_deviations(model, df: pd.DataFrame, device: torch.device,
                       seq_len: int = 64) -> np.ndarray:
    df_feat = add_features(df)

    # Normalise (use dataset statistics approximation)
    means = df_feat[FEATURE_COLS].mean()
    stds  = df_feat[FEATURE_COLS].std().replace(0, 1)
    features = ((df_feat[FEATURE_COLS] - means) / stds).values.astype(np.float32)

    n = len(features)
    deviations = np.zeros(n)
    counts     = np.zeros(n)

    with torch.no_grad():
        for i in range(0, n - seq_len + 1, seq_len // 2):
            chunk = features[i:i+seq_len]
            if len(chunk) < seq_len:
                break
            x = torch.tensor(chunk).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).cpu().numpy()
            deviations[i:i+seq_len] += pred
            counts[i:i+seq_len]     += 1

    counts = np.maximum(counts, 1)
    return deviations / counts


def apply_deviations(midi: pretty_midi.PrettyMIDI, df: pd.DataFrame,
                     deviations: np.ndarray, strength: float = 1.0) -> pretty_midi.PrettyMIDI:
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())
    inst_out = pretty_midi.Instrument(program=0)

    for idx, (_, row) in enumerate(df.iterrows()):
        dev_sec = (deviations[idx] * strength) / 1000.0
        new_start = max(0.0, row['onset'] + dev_sec)
        new_end   = new_start + row['duration']
        note = pretty_midi.Note(
            velocity=int(row['velocity']),
            pitch=int(row['pitch']),
            start=new_start,
            end=new_end,
        )
        inst_out.notes.append(note)

    inst_out.notes.sort(key=lambda n: n.start)
    new_midi.instruments.append(inst_out)
    return new_midi


def humanise(input_path: str, output_path: str, model_path: str,
             strength: float = 1.0, seq_len: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(model_path, input_dim=len(FEATURE_COLS), device=device)
    midi, df = midi_to_df(input_path)

    print(f"Loaded {len(df)} notes from {input_path}")
    deviations = predict_deviations(model, df, device, seq_len)

    print(f"Predicted deviations: mean={deviations.mean():.2f}ms std={deviations.std():.2f}ms")
    new_midi = apply_deviations(midi, df, deviations, strength)
    new_midi.write(output_path)
    print(f"Saved humanised MIDI → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--model",    required=True)
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Deviation strength multiplier (0.5=subtle, 1.0=full, 1.5=exaggerated)")
    args = parser.parse_args()
    humanise(args.input, args.output, args.model, args.strength)