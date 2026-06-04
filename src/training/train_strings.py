import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.data.strings_dataset import StringsDataset
from src.models.hybrid_strings import HybridTimingModel
from src.data.features import FEATURE_COLS


def train(
    data_path="data/processed/strings_train.parquet",
    val_path="data/processed/strings_val.parquet",
    output_dir="experiments",
    model_type="hybrid",
    seq_len=64,
    batch_size=128,
    hidden_dim=256,
    num_layers=2,
    lr=3e-4,
    epochs=60,
    patience=8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Strings model with pitch bend")

    train_df = pd.read_parquet(data_path)
    val_df   = pd.read_parquet(val_path)

    train_ds = StringsDataset(train_df, seq_len=seq_len)
    val_ds   = StringsDataset(val_df,   seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model     = HybridTimingModel(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion_timing = nn.HuberLoss(delta=0.5)
    criterion_pb     = nn.HuberLoss(delta=0.1)
    scaler    = torch.amp.GradScaler()

    best_val   = float("inf")
    no_improve = 0
    run_id     = f"strings_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    os.makedirs(output_dir, exist_ok=True)

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x)  # (batch, seq, 2)
                loss = (
                    1.0 * criterion_timing(pred[..., 0], y[..., 0]) +
                    0.5 * criterion_pb(pred[..., 1],     y[..., 1])
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss   = 0.0
        mae_timing = 0.0
        mae_pb     = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    pred = model(x)
                loss = (
                    1.0 * criterion_timing(pred[..., 0], y[..., 0]) +
                    0.5 * criterion_pb(pred[..., 1],     y[..., 1])
                )
                val_loss   += loss.item()
                mae_timing += (pred[..., 0] - y[..., 0]).abs().mean().item() * 200
                mae_pb     += (pred[..., 1] - y[..., 1]).abs().mean().item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        mae_timing /= len(val_loader)
        mae_pb     /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} MAE_timing={mae_timing:.2f}ms MAE_pb={mae_pb:.4f}")
        history.append({"epoch": epoch+1, "train": train_loss, "val": val_loss,
                        "mae_timing": mae_timing, "mae_pb": mae_pb})

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_strings_{run_id}.pt"))
            print("  Saved best model")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    with open(os.path.join(output_dir, f"history_strings_{run_id}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val: {best_val:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--epochs",  type=int,   default=60)
    parser.add_argument("--patience",type=int,   default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    train(lr=args.lr, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
