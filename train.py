"""
Training loop for the U-Net baseline.
Run from a terminal: python train.py
"""

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import time

from dataset import create_or_load_split, BraTSDataset
from models.unet_baseline import UNet3D
from losses import DiceBCELoss


ROOT_DIR = Path(r"D:\MMD\IPA Assignment\BraTS2021\BraTS2021_Training_Data")
SPLIT_DIR = Path(r"D:\MMD\IPA Assignment\BraTS2021\splits")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

PATCH_SIZE = 96
BATCH_SIZE = 2
NUM_EPOCHS = 100
LR = 1e-4
EARLY_STOP_PATIENCE = 15  # stop if val dice hasn't improved in this many epochs

QUICK_TEST = False  # set False before a real training run
QUICK_TEST_TRAIN_SIZE = 20
QUICK_TEST_VAL_SIZE = 6


def dice_score(logits, targets, threshold=0.5, eps=1e-5):
    """Hard dice, for monitoring during training. Not the loss, just a readable metric."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (0, 2, 3, 4)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)

    return ((2 * intersection + eps) / (union + eps)).mean().item()


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_dice = 0.0

    for i, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            with autocast("cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        total_dice += dice_score(logits, labels)

        mode = "train" if train else "val"
        print(f"  [{mode}] batch {i+1}/{len(loader)}  loss {loss.item():.4f}", flush=True)

    return total_loss / len(loader), total_dice / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids, val_ids, _ = create_or_load_split(ROOT_DIR, SPLIT_DIR)

    if QUICK_TEST:
        train_ids = train_ids[:QUICK_TEST_TRAIN_SIZE]
        val_ids = val_ids[:QUICK_TEST_VAL_SIZE]

    train_ds = BraTSDataset(ROOT_DIR, train_ids, patch_size=PATCH_SIZE, apply_dropout=True)
    val_ds = BraTSDataset(ROOT_DIR, val_ids, patch_size=PATCH_SIZE, apply_dropout=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet3D(in_channels=4, out_channels=3).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler("cuda")

    best_val_dice = 0.0
    epochs_since_improvement = 0
    num_epochs = 2 if QUICK_TEST else NUM_EPOCHS

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, loss_fn, optimizer, scaler, device, train=False)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f"epoch {epoch+1}/{num_epochs}  train_loss {train_loss:.4f}  train_dice {train_dice:.4f}  "
              f"val_loss {val_loss:.4f}  val_dice {val_dice:.4f}  time {epoch_time:.1f}s")

        torch.save(model.state_dict(), CHECKPOINT_DIR / "last.pt")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_since_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
        else:
            epochs_since_improvement += 1

        if not QUICK_TEST and epochs_since_improvement >= EARLY_STOP_PATIENCE:
            print(f"no improvement in {EARLY_STOP_PATIENCE} epochs, stopping early")
            break


if __name__ == "__main__":
    main()