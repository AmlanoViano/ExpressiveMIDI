import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.data.expression_dataset import ExpressionDataset
from src.models.expression import ExpressionModel
from src.data.features import FEATURE_COLS


def train(
    data_path="data/processed/train.parquet",
    val_path="data/processed/validation.parquet",
    output_dir="experiments",
    lr=3e-4,
    epochs=60,
    batch_size=128,
    patience=8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_parquet(data_path)
    val_df   = pd.read_parquet(val_path)

    train_ds = ExpressionDataset(train_df)
    val_ds   = ExpressionDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model     = ExpressionModel(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler    = torch.amp.GradScaler()

    criterion_vel     = nn.HuberLoss(delta=0.5)
    criterion_sustain = nn.BCEWithLogitsLoss()
    criterion_soft    = nn.BCEWithLogitsLoss()

    best_val   = float("inf")
    no_improve = 0
    run_id     = f"expression_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    os.makedirs(output_dir, exist_ok=True)

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x)
                loss = (
                    1.0 * criterion_vel(pred[..., 0],     y[..., 0]) +
                    0.5 * criterion_sustain(pred[..., 1], y[..., 1]) +
                    0.3 * criterion_soft(pred[..., 2],    y[..., 2])
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        mae_vel  = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    pred = model(x)
                loss = (
                    1.0 * criterion_vel(pred[..., 0],     y[..., 0]) +
                    0.5 * criterion_sustain(pred[..., 1], y[..., 1]) +
                    0.3 * criterion_soft(pred[..., 2],    y[..., 2])
                )
                val_loss += loss.item()
                mae_vel  += (pred[..., 0] - y[..., 0]).abs().mean().item() * 127

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        mae_vel    /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} MAE_vel={mae_vel:.2f}")
        history.append({"epoch": epoch+1, "train": train_loss, "val": val_loss, "mae_vel": mae_vel})

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_{run_id}.pt"))
            print("  Saved best model")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    with open(os.path.join(output_dir, f"history_{run_id}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val: {best_val:.4f}")


if __name__ == "__main__":
    train()
