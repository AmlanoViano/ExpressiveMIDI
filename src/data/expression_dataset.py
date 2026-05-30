import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .features import FEATURE_COLS, add_features

EXPRESSION_TARGET_COLS = ["velocity_dev", "sustain_pedal", "soft_pedal"]


class ExpressionDataset(Dataset):
    def __init__(self, df, seq_len=64, stride=32):
        df = add_features(df)
        means    = df[FEATURE_COLS].mean()
        stds     = df[FEATURE_COLS].std().replace(0, 1)
        features = ((df[FEATURE_COLS] - means) / stds).values.astype(np.float32)
        targets  = df[EXPRESSION_TARGET_COLS].values.astype(np.float32)
        self.samples = []
        for i in range(0, len(features) - seq_len, stride):
            self.samples.append((features[i:i+seq_len], targets[i:i+seq_len]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
