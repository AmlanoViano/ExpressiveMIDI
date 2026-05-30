import os
import pretty_midi
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_maestro_metadata(maestro_dir):
    return pd.read_csv(os.path.join(maestro_dir, "maestro-v3.0.0.csv"))


def quantise_onsets(onsets, resolution):
    return np.round(onsets / resolution) * resolution


def get_pedal_series(control_changes, cc_number, note_times):
    cc_filtered = [(cc.time, cc.value / 127.0) for cc in control_changes if cc.number == cc_number]
    if not cc_filtered:
        return np.zeros(len(note_times))
    times, vals = zip(*cc_filtered)
    times = np.array(times)
    vals  = np.array(vals)
    idx   = np.searchsorted(times, note_times, side="right") - 1
    return np.where(idx >= 0, vals[np.clip(idx, 0, len(vals)-1)], 0.0)


def extract_deviations_single(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
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
    if len(notes) < 10:
        return None
    df = pd.DataFrame(notes).sort_values("onset").reset_index(drop=True)
    note_times = df["onset"].values

    # Pedal extraction (vectorised)
    all_cc = []
    for inst in midi.instruments:
        if not inst.is_drum:
            all_cc.extend(inst.control_changes)
    all_cc = sorted(all_cc, key=lambda x: x.time)
    df["sustain_pedal"] = get_pedal_series(all_cc, 64, note_times)
    df["soft_pedal"]    = get_pedal_series(all_cc, 67, note_times)

    # Timing deviation
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


def augment_with_quantised(df, prob=0.3):
    if np.random.random() > prob:
        return df
    df = df.copy()
    ioi_mean  = df["local_ioi"].median()
    quantised = np.round(df["onset"].values / ioi_mean) * ioi_mean
    df["onset_score"]  = quantised
    df["deviation_ms"] = np.clip((df["onset"].values - quantised) * 1000.0, -200, 200)
    df["local_ioi"]    = ioi_mean
    return df


def process_dataset(maestro_dir, output_dir, split="train", max_files=None):
    meta     = load_maestro_metadata(maestro_dir)
    split_df = meta[meta["split"] == split].reset_index(drop=True)
    if max_files:
        split_df = split_df.iloc[:max_files]
    all_dfs = []
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split}"):
        path = os.path.join(maestro_dir, row["midi_filename"])
        try:
            df = extract_deviations_single(path)
            if df is None:
                continue
            df = augment_with_quantised(df, prob=0.3)
            df["piece_id"] = idx
            df["composer"] = row["canonical_composer"]
            df["title"]    = row["canonical_title"]
            df["year"]     = row["year"]
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["deviation_ms"] = combined["deviation_ms"].clip(-200, 200)
        combined["velocity_dev"] = combined["velocity_dev"].clip(-1.0, 1.0)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{split}.parquet")
        combined.to_parquet(out_path, index=False)
        print(f"Saved {len(combined)} notes -> {out_path}")
        print(f"Columns: {list(combined.columns)}")
        print(f"sustain_pedal mean: {combined['sustain_pedal'].mean():.2f}")
        print(f"soft_pedal mean:    {combined['soft_pedal'].mean():.2f}")
        return combined
    return None
