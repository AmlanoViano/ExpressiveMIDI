"""CNN + Transformer that maps score features to expressive performance parameters."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ConvBlock(nn.Module):
    def __init__(self, d, k, dropout):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k // 2)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B, L, D)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(torch.nn.functional.gelu(self.norm(y)))


class HumaniserNet(nn.Module):
    def __init__(self, n_features, n_targets=6, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.config = dict(n_features=n_features, n_targets=n_targets, d_model=d_model,
                           nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.inp = nn.Linear(n_features, d_model)
        self.convs = nn.Sequential(*[ConvBlock(d_model, k, dropout) for k in (3, 5, 7)])
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, n_targets))

    def forward(self, x, pad_mask=None):
        h = self.convs(self.inp(x))
        h = self.encoder(self.pos(h), src_key_padding_mask=pad_mask)
        return self.out(h)
