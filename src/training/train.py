"""
train.py - Training loop with GPU optimisation, early stopping.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.data.dataset import NoteSequenceDataset
from src.models.baseline import LSTMBaseline
from src.data.features import FEATURE_COLS


def train(
    data_path: str,
    val_path: str = "data/processed/validation.parquet",
    output_dir: str = "experiments",
    model_type: str = "lstm",
    seq_len: int = 64,
    batch_size: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    lr: float = 1e-3,
    epochs: int = 60,
    patience: int = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Model: {model_type}")

    train_df = pd.read_parquet(data_path)
    val_df   = pd.read_parquet(val_path)

    train_ds = NoteSequenceDataset(train_df, seq_len=seq_len)
    val_ds   = NoteSequenceDataset(val_df,   seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    if model_type == "transformer":
        from src.models.transformer import TimingTransformer
        model = TimingTransformer(input_dim=len(FEATURE_COLS)).to(device)
    elif model_type == "hybrid":
        from src.models.hybrid import HybridTimingModel
        model = HybridTimingModel(input_dim=len(FEATURE_COLS)).to(device)
    else:
        model = LSTMBaseline(input_dim=len(FEATURE_COLS), hidden_dim=hidden_dim,
                             num_layers=num_layers, dropout=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.HuberLoss(delta=20.0)
    scaler    = torch.amp.GradScaler()

    best_val_loss  = float('inf')
    epochs_no_improve = 0
    run_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    history = []
    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0.0
        mae      = 0.0
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
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train={train_loss:.2f} val={val_loss:.2f} MAE={mae:.2f}ms")
        history.append({'epoch': epoch+1, 'train_loss': train_loss,
                        'val_loss': val_loss, 'mae_ms': mae})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f"best_{run_id}.pt"))
            print("  ✓ Saved best model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    with open(os.path.join(output_dir, f"history_{run_id}.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.2f}")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer", "hybrid"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    train(
        data_path="data/processed/train.parquet",
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )