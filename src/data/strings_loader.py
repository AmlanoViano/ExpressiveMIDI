import os
import pretty_midi
import numpy as np
import pandas as pd
from tqdm import tqdm


def quantise_onsets(onsets, resolution):
    return np.round(onsets / resolution) * resolution


def extract_strings_single(midi_path):
    """Extract notes from all tracks, merge, compute timing deviations."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    pitch_bends_all = []

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({
                "pitch":    n.pitch,
                "onset":    n.start,
                "duration": n.end - n.start,
                "velocity": n.velocity,
                "offset":   n.end,
            })
        pitch_bends_all.extend(inst.pitch_bends)

    if len(notes) < 10:
        return None

    df = pd.DataFrame(notes).sort_values("onset").reset_index(drop=True)
    note_times = df["onset"].values

    # Pitch bend at each note onset (vibrato/portamento)
    pb_sorted = sorted(pitch_bends_all, key=lambda x: x.time)
    pb_times  = np.array([pb.time for pb in pb_sorted])
    pb_vals   = np.array([pb.pitch / 8192.0 for pb in pb_sorted])  # normalise -1 to 1

    if len(pb_times) > 0:
        idx = np.searchsorted(pb_times, note_times, side="right") - 1
        pitch_bend = np.where(idx >= 0, pb_vals[np.clip(idx, 0, len(pb_vals)-1)], 0.0)
    else:
        pitch_bend = np.zeros(len(note_times))

    df["pitch_bend"] = pitch_bend

    # Timing deviation via adaptive quantisation
    ioi = np.diff(note_times, prepend=note_times[0])
    ioi = np.clip(ioi, 0.02, 2.0)
    local_ioi = pd.Series(ioi).rolling(16, center=True, min_periods=1).median().values
    local_ioi = np.clip(local_ioi, 0.05, 1.0)
    quantised    = quantise_onsets(note_times, resolution=local_ioi.mean() / 2)
    deviation_ms = (note_times - quantised) * 1000.0
    vel_mean     = pd.Series(df["velocity"].values).rolling(16, center=True, min_periods=1).mean().values
    velocity_dev = (df["velocity"].values - vel_mean) / 127.0

    df["onset_score"]    = quantised
    df["duration_score"] = df["duration"]
    df["velocity_score"] = vel_mean.astype(int)
    df["deviation_ms"]   = deviation_ms
    df["velocity_dev"]   = velocity_dev
    df["local_ioi"]      = local_ioi
    return df


def process_strings_dataset(strings_dir, output_dir, max_files=None):
    midi_files = []
    for root, dirs, files in os.walk(strings_dir):
        for f in files:
            if f.endswith(".mid"):
                midi_files.append(os.path.join(root, f))

    if max_files:
        midi_files = midi_files[:max_files]

    print(f"Found {len(midi_files)} MIDI files")

    all_dfs = []
    for idx, path in enumerate(tqdm(midi_files, desc="Processing strings")):
        try:
            df = extract_strings_single(path)
            if df is None:
                continue
            df["piece_id"] = idx
            df["filename"] = os.path.basename(path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["deviation_ms"] = combined["deviation_ms"].clip(-200, 200)
        combined["velocity_dev"] = combined["velocity_dev"].clip(-1.0, 1.0)
        os.makedirs(output_dir, exist_ok=True)

        # 80/20 train/val split
        pieces     = combined["piece_id"].unique()
        np.random.seed(42)
        np.random.shuffle(pieces)
        split      = int(len(pieces) * 0.8)
        train_ids  = pieces[:split]
        val_ids    = pieces[split:]

        train_df = combined[combined["piece_id"].isin(train_ids)]
        val_df   = combined[combined["piece_id"].isin(val_ids)]

        train_df.to_parquet(os.path.join(output_dir, "strings_train.parquet"), index=False)
        val_df.to_parquet(os.path.join(output_dir, "strings_val.parquet"), index=False)

        print(f"Train: {len(train_df)} notes | Val: {len(val_df)} notes")
        print(f"deviation_ms std: {combined['deviation_ms'].std():.2f}")
        print(f"pitch_bend std:   {combined['pitch_bend'].std():.4f}")
        return combined
    return None
