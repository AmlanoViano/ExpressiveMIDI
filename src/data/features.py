"""
features.py - Rich note-level feature engineering for timing prediction.
"""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    # --- Pitch features ---
    df['pitch_class'] = df['pitch'] % 12
    df['octave']      = df['pitch'] // 12

    # --- Interval features ---
    df['interval_prev'] = df['pitch'].diff().fillna(0).clip(-24, 24)
    df['interval_next'] = df['pitch'].diff(-1).fillna(0).clip(-24, 24)
    df['interval_abs']  = df['interval_prev'].abs()

    # --- IOI (inter-onset interval) ---
    ioi = df['onset_score'].diff().fillna(0.5).clip(0.02, 4.0)
    df['local_ioi'] = ioi

    # Rolling median IOI = local tempo reference
    df['tempo_ref'] = pd.Series(ioi).rolling(8, center=True, min_periods=1).median().values
    df['tempo_ref'] = df['tempo_ref'].clip(0.05, 2.0)

    # IOI relative to local tempo (1.0 = on tempo, >1 = slower)
    df['ioi_ratio'] = df['local_ioi'] / df['tempo_ref']

    # --- Beat position from tempo reference ---
    # Cumulative beat count
    beat_pos_raw = (df['onset_score'] / df['tempo_ref'].mean()).values
    df['beat_pos_cont'] = beat_pos_raw % 4          # continuous 0-4
    df['beat_pos']      = beat_pos_raw.astype(int) % 4  # discrete 0-3

    # Beat strength: downbeat > mid > offbeat
    beat_strength_map = {0: 1.0, 2: 0.6, 1: 0.3, 3: 0.2}
    df['beat_strength'] = df['beat_pos'].map(beat_strength_map).fillna(0.2)

    # --- Duration features ---
    df['duration_norm']  = df['duration_score'].clip(0, 4) / 4.0
    df['duration_ratio'] = df['duration_score'] / (df['tempo_ref'] + 1e-6)

    # --- Velocity features ---
    df['velocity_norm'] = df['velocity_score'] / 127.0
    # Local velocity context (deviation from rolling mean)
    vel_mean = pd.Series(df['velocity_score']).rolling(8, center=True, min_periods=1).mean()
    df['velocity_rel'] = (df['velocity_score'] - vel_mean) / 127.0

    # --- Chord detection (notes within 50ms of each other = chord) ---
    onset_diff = df['onset_score'].diff().abs().fillna(0)
    df['in_chord'] = (onset_diff < 0.05).astype(float)
    # Chord size (how many notes in local 50ms window)
    df['chord_size'] = df['in_chord'].rolling(4, center=True, min_periods=1).sum() / 4.0

    # --- Phrase position (local, 16-note window) ---
    n = len(df)
    phrase_len = 16
    df['phrase_pos'] = (np.arange(n) % phrase_len) / (phrase_len - 1)

    # --- Phrase boundary (last 2 notes of phrase = 1) ---
    df['phrase_end'] = ((np.arange(n) % phrase_len) >= (phrase_len - 2)).astype(float)
    df['phrase_start'] = ((np.arange(n) % phrase_len) < 2).astype(float)

    # --- Melodic contour (direction of melody) ---
    df['contour'] = np.sign(df['interval_prev'])

    # --- Register (normalized pitch) ---
    df['pitch_norm'] = df['pitch'] / 127.0

    return df


FEATURE_COLS = [
    # Pitch
    'pitch_norm', 'pitch_class', 'octave',
    # Intervals
    'interval_prev', 'interval_next', 'interval_abs', 'contour',
    # Timing
    'local_ioi', 'tempo_ref', 'ioi_ratio', 'beat_pos_cont', 'beat_strength',
    # Duration
    'duration_norm', 'duration_ratio',
    # Velocity
    'velocity_norm', 'velocity_rel',
    # Chord
    'in_chord', 'chord_size',
    # Phrase
    'phrase_pos', 'phrase_end', 'phrase_start',
]

TARGET_COL = 'deviation_ms'