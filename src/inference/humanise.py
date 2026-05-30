import argparse
import torch
import numpy as np
import pretty_midi
import pandas as pd

from src.data.features import add_features, FEATURE_COLS
from src.models.hybrid import HybridTimingModel
from src.models.expression import ExpressionModel


def load_timing_model(model_path, input_dim, device):
    model = HybridTimingModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_expression_model(model_path, input_dim, device):
    model = ExpressionModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def midi_to_df(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                notes.append({
                    "pitch": n.pitch, "onset": n.start,
                    "onset_score": n.start, "duration": n.end - n.start,
                    "duration_score": n.end - n.start, "velocity": n.velocity,
                    "velocity_score": n.velocity, "offset": n.end,
                    "sustain_pedal": 0.0, "soft_pedal": 0.0,
                })
    df = pd.DataFrame(notes).sort_values("onset").reset_index(drop=True)
    return midi, df


def predict(model, df, device, seq_len=64):
    df_feat  = add_features(df)
    means    = df_feat[FEATURE_COLS].mean()
    stds     = df_feat[FEATURE_COLS].std().replace(0, 1)
    features = ((df_feat[FEATURE_COLS] - means) / stds).values.astype(np.float32)
    n        = len(features)
    seq_len  = min(seq_len, n)
    stride   = max(1, seq_len // 2)
    is_multi = False
    with torch.no_grad():
        test_x = torch.tensor(features[:seq_len]).unsqueeze(0).to(device)
        test_o = model(test_x)
        is_multi = test_o.dim() == 3
    if is_multi:
        n_out = test_o.shape[-1]
        preds  = np.zeros((n, n_out))
    else:
        preds  = np.zeros(n)
    counts = np.zeros(n)
    with torch.no_grad():
        for i in range(0, n, stride):
            chunk = features[i:i+seq_len]
            if len(chunk) < seq_len:
                chunk = np.pad(chunk, ((0, seq_len-len(chunk)), (0,0)))
            x   = torch.tensor(chunk).unsqueeze(0).to(device)
            out = model(x).squeeze(0).cpu().numpy()
            end = min(i+seq_len, n)
            length = end - i
            if is_multi:
                preds[i:end]  += out[:length]
            else:
                preds[i:end]  += out[:length]
            counts[i:end] += 1
    counts = np.maximum(counts, 1)
    if is_multi:
        return preds / counts[:, None]
    return preds / counts


def humanise(input_path, output_path, timing_model_path, expression_model_path=None, strength=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    timing_model = load_timing_model(timing_model_path, len(FEATURE_COLS), device)
    midi, df     = midi_to_df(input_path)
    print(f"Loaded {len(df)} notes")

    # Timing predictions
    timing_devs = predict(timing_model, df, device)
    timing_devs -= timing_devs.mean()
    print(f"Timing - mean={timing_devs.mean():.2f}ms std={timing_devs.std():.2f}ms")

    # Expression predictions
    expr_preds = None
    if expression_model_path:
        expr_model = load_expression_model(expression_model_path, len(FEATURE_COLS), device)
        expr_preds = predict(expr_model, df, device)
        print(f"Velocity MAE approx std={expr_preds[:,0].std():.3f}")
        print(f"Sustain mean={torch.sigmoid(torch.tensor(expr_preds[:,1])).mean():.2f}")
        print(f"Soft mean={torch.sigmoid(torch.tensor(expr_preds[:,2])).mean():.2f}")

    # Build output MIDI
    tempo_times, tempos = midi.get_tempo_changes()
    initial_tempo = tempos[0] if len(tempos) > 0 else 120.0
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    inst_out = pretty_midi.Instrument(program=0)

    for idx, (_, row) in enumerate(df.iterrows()):
        dev_sec   = (timing_devs[idx] * strength) / 1000.0
        new_start = max(0.0, row["onset"] + dev_sec)
        new_end   = new_start + row["duration"]
        new_vel   = int(row["velocity"])
        if expr_preds is not None:
            vel_dev = expr_preds[idx, 0] * strength
            new_vel = int(np.clip(row["velocity"] + vel_dev * 127, 1, 127))
        inst_out.notes.append(pretty_midi.Note(
            velocity=new_vel, pitch=int(row["pitch"]),
            start=new_start, end=new_end
        ))

    # Add pedal CC events
    if expr_preds is not None:
        for idx, (_, row) in enumerate(df.iterrows()):
            sustain_val = int(np.clip(torch.sigmoid(torch.tensor(expr_preds[idx, 1])).item() * 127, 0, 127))
            soft_val    = int(np.clip(torch.sigmoid(torch.tensor(expr_preds[idx, 2])).item() * 127, 0, 127))
            inst_out.control_changes.append(pretty_midi.ControlChange(64, sustain_val, df.iloc[idx]["onset"]))
            inst_out.control_changes.append(pretty_midi.ControlChange(67, soft_val,    df.iloc[idx]["onset"]))
        inst_out.control_changes.sort(key=lambda c: c.time)

    inst_out.notes.sort(key=lambda n: n.start)
    new_midi.instruments.append(inst_out)
    new_midi.write(output_path)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--model",      required=True, help="Timing model path")
    parser.add_argument("--expr_model", default=None,  help="Expression model path (optional)")
    parser.add_argument("--strength",   type=float, default=1.0)
    args = parser.parse_args()
    humanise(args.input, args.output, args.model, args.expr_model, args.strength)
