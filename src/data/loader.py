"""
loader.py - Extract expressive timing deviations from single MAESTRO performances.
Method: quantise each performance to a grid, deviation = actual - quantised onset.
This is the standard approach in expressive performance research.
"""
import os
import pretty_midi
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_maestro_metadata(maestro_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(maestro_dir, "maestro-v3.0.0.csv")
    return pd.read_csv(csv_path)


def quantise_onsets(onsets: np.ndarray, resolution: float = 0.125) -> np.ndarray:
    """
    Snap each onset to the nearest grid point.
    resolution: grid size in seconds (0.125s = 8th note at 60bpm)
    Returns quantised onsets.
    """
    return np.round(onsets / resolution) * resolution


def extract_deviations_single(midi_path: str, grid_resolution: float = 0.05) -> pd.DataFrame:
    """
    Extract notes from a single performance.
    Estimate 'score' onsets by local tempo-adaptive quantisation.
    Deviation = performed onset - quantised onset.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                notes.append({
                    'pitch':    n.pitch,
                    'onset':    n.start,
                    'duration': n.end - n.start,
                    'velocity': n.velocity,
                    'offset':   n.end,
                })

    if len(notes) < 10:
        return None

    df = pd.DataFrame(notes).sort_values('onset').reset_index(drop=True)

    # Estimate local tempo from inter-onset intervals
    ioi = np.diff(df['onset'].values, prepend=df['onset'].values[0])
    ioi = np.clip(ioi, 0.02, 2.0)

    # Adaptive grid: use median IOI in local window as beat estimate
    window = 16
    local_ioi = pd.Series(ioi).rolling(window, center=True, min_periods=1).median().values
    local_ioi = np.clip(local_ioi, 0.05, 1.0)

    # Quantise to nearest local beat subdivision
    quantised = quantise_onsets(df['onset'].values, resolution=local_ioi.mean() / 2)
    deviation_ms = (df['onset'].values - quantised) * 1000.0

    df['onset_score']   = quantised
    df['deviation_ms']  = deviation_ms
    df['local_ioi']     = local_ioi
    df['velocity_score'] = df['velocity']
    df['duration_score'] = df['duration']

    return df


def process_dataset(maestro_dir: str, output_dir: str,
                    split: str = 'train', max_files: int = None):
    meta = load_maestro_metadata(maestro_dir)
    split_df = meta[meta['split'] == split].reset_index(drop=True)

    if max_files:
        split_df = split_df.iloc[:max_files]

    all_dfs = []
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split}"):
        path = os.path.join(maestro_dir, row['midi_filename'])
        try:
            df = extract_deviations_single(path)
            if df is None:
                continue
            df = augment_with_quantised(df, prob=0.3)
            df['piece_id'] = idx
            df['composer'] = row['canonical_composer']
            df['title']    = row['canonical_title']
            df['year']     = row['year']
            all_dfs.append(df)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined['deviation_ms'] = combined['deviation_ms'].clip(-200, 200)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{split}.parquet")
        combined.to_parquet(out_path, index=False)
        print(f"Saved {len(combined)} notes → {out_path}")
        p = combined['deviation_ms']
        print(f"Deviation stats (ms): mean={p.mean():.2f} std={p.std():.2f}")
        print(f"Percentiles: 5%={p.quantile(0.05):.1f} 25%={p.quantile(0.25):.1f} "
              f"median={p.median():.1f} 75%={p.quantile(0.75):.1f} 95%={p.quantile(0.95):.1f}")
        return combined
    return None
 

def augment_with_quantised(df: pd.DataFrame, prob: float = 0.3) -> pd.DataFrame:
    """
    With probability prob, replace onset_score with perfectly quantised onsets.
    This teaches the model to humanise grid-aligned input.
    """
    if np.random.random() > prob:
        return df
    df = df.copy()
    ioi_mean = df["local_ioi"].median()
    quantised = np.round(df["onset"].values / ioi_mean) * ioi_mean
    df["onset_score"] = quantised
    df["deviation_ms"] = (df["onset"].values - quantised) * 1000.0
    df["deviation_ms"] = df["deviation_ms"].clip(-200, 200)
    df["local_ioi"] = ioi_mean
    return df
