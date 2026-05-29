"""
features.py - Engineer note-level features from deviation DataFrame.
"""
import numpy as np
import pandas as pd


BEAT_STRENGTHS = {0: 1.0, 2: 0.75, 1: 0.5, 3: 0.25}  # beat position mod 4


def add_features(df: pd.DataFrame, ticks_per_beat: float = 0.5) -> pd.DataFrame:
    df = df.copy()

    # Beat position (0-3) based on score onset
    df['beat_pos']      = (df['onset_score'] / ticks_per_beat).astype(int) % 4
    df['beat_strength'] = df['beat_pos'].map(BEAT_STRENGTHS).fillna(0.25)

    # Pitch features
    df['pitch_class']   = df['pitch'] % 12
    df['octave']        = df['pitch'] // 12

    # Interval to previous/next note
    df['interval_prev'] = df['pitch'].diff().fillna(0)
    df['interval_next'] = df['pitch'].diff(-1).fillna(0)

    # Local tempo estimate (beats per second from consecutive onsets)
    onset_diffs = df['onset_score'].diff().fillna(0.5).clip(0.05, 5.0)
    df['local_ioi']     = onset_diffs  # inter-onset interval in seconds

    # Normalised duration
    df['duration_norm'] = df['duration_score'].clip(0, 4) / 4.0

    # Velocity normalised
    df['velocity_norm'] = df['velocity_score'] / 127.0

    # Phrase position (position within a rolling 8-note window)
    df['phrase_pos']    = np.tile(np.arange(8), len(df) // 8 + 1)[:len(df)] / 7.0

    # Clip deviation target to ±500ms (outlier removal)
    df['deviation_ms']  = df['deviation_ms'].clip(-500, 500)

    return df


FEATURE_COLS = [
    'pitch', 'pitch_class', 'octave',
    'beat_pos', 'beat_strength',
    'interval_prev', 'interval_next',
    'local_ioi', 'duration_norm', 'velocity_norm',
    'phrase_pos', 'duration_score',
]

TARGET_COL = 'deviation_ms'