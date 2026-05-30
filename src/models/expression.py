import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ExpressionModel(nn.Module):
    """
    Predicts 3 expressive targets per note:
      - velocity_dev: deviation from mean velocity (-1 to 1)
      - sustain_pedal: CC64 value (0 to 1)
      - soft_pedal: CC67 value (0 to 1)
    """
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_velocity = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        self.head_sustain  = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        self.head_soft     = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        vel     = self.head_velocity(x).squeeze(-1)
        sustain = self.head_sustain(x).squeeze(-1)
        soft    = self.head_soft(x).squeeze(-1)
        return torch.stack([vel, sustain, soft], dim=-1)  # (batch, seq, 3)
