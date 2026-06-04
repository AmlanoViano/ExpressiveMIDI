import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pretty_midi
from tqdm import tqdm
from datetime import datetime

from src.models.hybrid import HybridTimingModel
from src.data.features import FEATURE_COLS, add_features
from src.data.dataset import NoteSequenceDataset
from src.data.loader import extract_deviations_single


def load_user_midis(midi_paths: list) -> pd.DataFrame:
    """Extract deviations from user-provided MIDI files."""
    all_dfs = []
    for i, path in enumerate(midi_paths):
        print(f"Processing {path}...")
        try:
            df = extract_deviations_single(path)
            if df is None:
                print(f"  Skipping - too few notes")
                continue
            df["piece_id"] = i
            df["composer"]  = "user"
            df["title"]     = os.path.basename(path)
            df["year"]      = 2024
            all_dfs.append(df)
            print(f"  {len(df)} notes extracted")
        except Exception as e:
            print(f"  Error: {e}")
    if not all_dfs:
        raise ValueError("No valid MIDI files found")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["deviation_ms"] = combined["deviation_ms"].clip(-200, 200)
    print(f"Total: {len(combined)} notes from {len(all_dfs)} files")
    return combined


def finetune(
    midi_paths: list,
    pretrained_model_path: str,
    style_name: str,
    output_dir: str = "experiments",
    lr: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 32,
    seq_len: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fine-tuning style: {style_name}")

    # Load user data
    df = load_user_midis(midi_paths)

    if len(df) < seq_len * 2:
        print(f"WARNING: Very few notes ({len(df)}). Results may be poor.")

    dataset = NoteSequenceDataset(df, seq_len=seq_len, stride=16)
    if len(dataset) == 0:
        raise ValueError("Not enough data to create sequences.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    # Load pretrained model
    model = HybridTimingModel(input_dim=len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print(f"Loaded pretrained model from {pretrained_model_path}")

    # Fine-tune with lower lr to preserve learned features
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=20.0)
    scaler    = torch.amp.GradScaler()

    best_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)
    out_path  = os.path.join(output_dir, f"style_{style_name}.pt")

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")
        history.append({"epoch": epoch+1, "loss": epoch_loss})

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_path)
            print(f"  Saved -> {out_path}")

    with open(os.path.join(output_dir, f"history_style_{style_name}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Style model saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--midis",     nargs="+", required=True, help="3-5 MIDI files of your playing")
    parser.add_argument("--name",      required=True, help="Name for this style (e.g. john)")
    parser.add_argument("--pretrained",required=True, help="Path to pretrained hybrid model")
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--epochs",    type=int,   default=20)
    args = parser.parse_args()
    finetune(
        midi_paths=args.midis,
        pretrained_model_path=args.pretrained,
        style_name=args.name,
        lr=args.lr,
        epochs=args.epochs,
    )
