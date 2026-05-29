"""
train.py - Training loop with GPU optimisation.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.data.dataset import NoteSequenceDataset
from src.models.baseline import LSTMBaseline
from src.data.features import FEATURE_COLS


def train(
    data_path: str,
    output_dir: str = "experiments",
    seq_len: int = 64,
    batch_size: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    lr: float = 1e-3,
    epochs: int = 30,
    val_split: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_parquet(data_path)
    dataset = NoteSequenceDataset(df, seq_len=seq_len)

    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = LSTMBaseline(input_dim=len(FEATURE_COLS), hidden_dim=hidden_dim,
                         num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=20.0)  # robust to outlier deviations

    # Mixed precision for speed
    scaler = torch.amp.GradScaler()

    best_val_loss = float('inf')
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    history = []
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    pred = model(x)
                val_loss += criterion(pred, y).item()
                mae      += (pred - y).abs().mean().item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        mae        /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}: train={train_loss:.2f} val={val_loss:.2f} MAE={mae:.2f}ms")
        history.append({'epoch': epoch+1, 'train_loss': train_loss,
                        'val_loss': val_loss, 'mae_ms': mae})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_{run_id}.pt"))
            print("  ✓ Saved best model")

    # Save history
    with open(os.path.join(output_dir, f"history_{run_id}.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.2f}")
    return history


if __name__ == "__main__":
    train(data_path="data/processed/train.parquet")