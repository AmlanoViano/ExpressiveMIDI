"""
dataset.py - PyTorch Dataset for note sequences.
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .features import FEATURE_COLS, TARGET_COL, add_features


class NoteSequenceDataset(Dataset):
    """
    Sliding window over note sequences.
    Each sample: (seq_len, n_features) -> (seq_len,) deviations
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 64, stride: int = 32):
        df = add_features(df)
        self.seq_len = seq_len

        # Normalise features
        self.feature_means = df[FEATURE_COLS].mean()
        self.feature_stds  = df[FEATURE_COLS].std().replace(0, 1)

        features = ((df[FEATURE_COLS] - self.feature_means) / self.feature_stds).values.astype(np.float32)
        targets  = df[TARGET_COL].values.astype(np.float32)

        self.samples = []
        for i in range(0, len(features) - seq_len, stride):
            x = features[i:i+seq_len]
            y = targets[i:i+seq_len]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)