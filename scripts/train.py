"""CLI entry point: train the U-Net (Stage 3).

Usage:
    python scripts/train.py                 # full run, config defaults
    python scripts/train.py --epochs 1 --limit-train 8 --limit-val 4   # quick test

Trains RGB patches -> normalised destriped EI on skin (masked L1 loss). Each epoch
draws CROPS_PER_IMAGE random crops per train image; validation predicts whole images
by tiling and reports masked MAE/MSE on skin. Keeps the best model by validation MAE
and stops early after EARLY_STOP_PATIENCE epochs without improvement. Writes the best
checkpoint and a per-epoch history CSV to OUTPUT_DIR. Run after the preprocessing
pipeline (masks + destriped EI + norm_stats.json).
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from src.dataset import ErythemaDataset, worker_init_fn
from src.inference import tiled_predict
from src.losses import masked_l1_loss
from src.metrics import masked_mae, masked_mse
from src.model import build_unet, get_device
from src.normalization import load_stats


def parse_args() -> argparse.Namespace:
    """Parse training arguments (defaults come from config)."""
    p = argparse.ArgumentParser(description="Train the erythema U-Net (Stage 3).")
    p.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--crops-per-image", type=int, default=config.CROPS_PER_IMAGE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--limit-train", type=int, default=None,
                   help="Use only the first N train images (smoke test).")
    p.add_argument("--limit-val", type=int, default=None,
                   help="Use only the first N val images (smoke test).")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device) -> float:
    """Run one training epoch; return the mean masked-L1 loss over batches."""
    model.train()
    total, n = 0.0, 0
    for rgb, ei, mask in loader:
        rgb, ei, mask = rgb.to(device), ei.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = masked_l1_loss(model(rgb), ei, mask)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, dataset, device) -> tuple:
    """Tile-predict every validation image; return dataset-level (MAE, MSE) on skin."""
    model.eval()
    abs_sum = sq_sum = px_sum = 0.0
    for i in range(len(dataset)):
        rgb, ei, mask = dataset[i]
        pred = tiled_predict(model, rgb, device)
        m = mask
        abs_sum += float(((pred - ei).abs() * m).sum())
        sq_sum += float(((pred - ei).pow(2) * m).sum())
        px_sum += float(m.sum())
    px_sum = max(px_sum, 1e-6)
    return abs_sum / px_sum, sq_sum / px_sum


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    processed = Path(config.LOCAL_PROCESSED_DIR)
    ei_dir = processed / "ei_maps_destriped"
    mask_dir = processed / "masks"
    stats = load_stats(str(processed / "norm_stats.json"))
    manifest = pd.read_csv(processed / "manifest.csv")

    train_rows = manifest[manifest["split"] == "train"]
    val_rows = manifest[manifest["split"] == "valid"]
    if args.limit_train:
        train_rows = train_rows.head(args.limit_train)
    if args.limit_val:
        val_rows = val_rows.head(args.limit_val)

    train_ds = ErythemaDataset(train_rows, ei_dir, mask_dir, stats, mode="train",
                               crops_per_image=args.crops_per_image, seed=args.seed)
    val_ds = ErythemaDataset(val_rows, ei_dir, mask_dir, stats, mode="full")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn, drop_last=True)

    device = get_device()
    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"
    history_path = out_dir / "train_history.csv"

    print(f"device={device}  train_imgs={len(train_rows)} x{args.crops_per_image} "
          f"crops = {len(train_ds)} samples  val_imgs={len(val_rows)}")

    best_mae = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae, val_mse = validate(model, val_ds, device)
        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_mae": val_mae, "val_mse": val_mse,
                        "best_mae": best_mae})
        print(f"epoch {epoch:3d}  train_L1={train_loss:.5f}  "
              f"val_MAE={val_mae:.5f}  val_MSE={val_mse:.5f}"
              f"{'  * best' if improved else ''}")

        if epochs_no_improve >= args.patience:
            print(f"Early stop: no val improvement for {args.patience} epochs.")
            break

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_mae",
                                               "val_mse", "best_mae"])
        writer.writeheader()
        writer.writerows(history)

    print(f"\nDone. Best val MAE={best_mae:.5f}  ->  {ckpt_path}\nHistory -> {history_path}")


if __name__ == "__main__":
    main()