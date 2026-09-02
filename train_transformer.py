"""
Training script for the MaskedFusionTransformer.

Key differences from train.py (baseline):
  1. Passes presence_mask to the model (critical — the transformer uses it for masking)
  2. Gradient accumulation (ACCUMULATE_STEPS=4) — transformer is ~4× more memory-hungry
     than the baseline U-Net at the same patch size, so we accumulate to maintain
     an effective batch size of 4 with real batch_size=1.
  3. Linear LR warmup for the first WARMUP_EPOCHS epochs, then cosine decay.
     Warmup is important for transformers: without it, large early gradients can
     destroy the attention projections before they settle into a useful regime.
  4. Slightly lower base LR than the U-Net (1e-4 → 5e-5) — transformer loss
     landscape is more sensitive to LR.

Run: python train_transformer.py
"""

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
import math

from dataset import create_or_load_split, BraTSDataset
from models.full_model import MaskedFusionTransformer
from losses import DiceBCELoss


# ── Config ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(r"D:\MMD\IPA Assignment\BraTS2021\BraTS2021_Training_Data")
SPLIT_DIR      = Path(r"D:\MMD\IPA Assignment\BraTS2021\splits")
CHECKPOINT_DIR = Path("checkpoints_transformer")
CHECKPOINT_DIR.mkdir(exist_ok=True)

PATCH_SIZE       = 96
BATCH_SIZE       = 1     # real batch size — kept at 1 due to transformer memory
ACCUMULATE_STEPS = 4     # effective batch size = BATCH_SIZE × ACCUMULATE_STEPS = 4
NUM_EPOCHS       = 120
BASE_LR          = 5e-5
WARMUP_EPOCHS    = 10
EARLY_STOP_PATIENCE = 20

QUICK_TEST            = False
QUICK_TEST_TRAIN_SIZE = 10
QUICK_TEST_VAL_SIZE   = 4
# ────────────────────────────────────────────────────────────────────────────


def dice_score(logits, targets, threshold=0.5, eps=1e-5):
    """Hard Dice — for monitoring only, not the training loss."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims  = (0, 2, 3, 4)
    inter = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)
    return ((2 * inter + eps) / (union + eps)).mean().item()


def get_lr(epoch: int, base_lr: float, warmup_epochs: int, total_epochs: int) -> float:
    """
    Linear warmup then cosine annealing.
    Returns a multiplier to apply to base_lr.
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, train, accumulate_steps):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_dice = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        images       = batch["image"].to(device)
        labels       = batch["label"].to(device)
        presence     = batch["presence_mask"].to(device)  # (B, 4) — critical

        with torch.set_grad_enabled(train):
            with autocast("cuda"):
                logits = model(images, presence)           # pass presence_mask
                loss   = loss_fn(logits, labels)
                if train:
                    loss_scaled = loss / accumulate_steps  # normalise for accumulation

            if train:
                scaler.scale(loss_scaled).backward()
                # Only step the optimiser every accumulate_steps batches
                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

        total_loss += loss.item()
        total_dice += dice_score(logits, labels)

        mode = "train" if train else "val"
        print(
            f"  [{mode}] batch {i+1}/{len(loader)}  loss {loss.item():.4f}",
            flush=True,
        )

    return total_loss / len(loader), total_dice / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids, val_ids, _ = create_or_load_split(ROOT_DIR, SPLIT_DIR)
    if QUICK_TEST:
        train_ids = train_ids[:QUICK_TEST_TRAIN_SIZE]
        val_ids   = val_ids[:QUICK_TEST_VAL_SIZE]

    train_ds = BraTSDataset(ROOT_DIR, train_ids, patch_size=PATCH_SIZE, apply_dropout=True)
    val_ds   = BraTSDataset(ROOT_DIR, val_ids,   patch_size=PATCH_SIZE, apply_dropout=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model   = MaskedFusionTransformer().to(device)
    loss_fn = DiceBCELoss()
    scaler  = GradScaler("cuda")

    # All parameters — stems, transformer, decoder — with the same optimiser.
    # Using AdamW with weight decay for mild L2 regularisation on the transformer
    # weights (attention projections and FFN linear layers tend to benefit from this).
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"Device: {device}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATE_STEPS}")
    print()

    best_val_dice = 0.0
    epochs_since_improvement = 0
    num_epochs = 2 if QUICK_TEST else NUM_EPOCHS

    for epoch in range(num_epochs):
        # Manually set LR (warmup + cosine) instead of using a scheduler,
        # for clarity and full control
        lr = get_lr(epoch, BASE_LR, WARMUP_EPOCHS, num_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        epoch_start = time.time()
        train_loss, train_dice = run_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device,
            train=True, accumulate_steps=ACCUMULATE_STEPS,
        )
        val_loss, val_dice = run_epoch(
            model, val_loader, loss_fn, optimizer, scaler, device,
            train=False, accumulate_steps=1,
        )
        epoch_time = time.time() - epoch_start

        print(
            f"epoch {epoch+1:3d}/{num_epochs}  lr {lr:.2e}  "
            f"train_loss {train_loss:.4f}  train_dice {train_dice:.4f}  "
            f"val_loss {val_loss:.4f}  val_dice {val_dice:.4f}  "
            f"time {epoch_time:.1f}s"
        )

        torch.save(model.state_dict(), CHECKPOINT_DIR / "last.pt")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_since_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
            print(f"  ✓ new best val_dice: {best_val_dice:.4f}")
        else:
            epochs_since_improvement += 1

        if not QUICK_TEST and epochs_since_improvement >= EARLY_STOP_PATIENCE:
            print(f"No improvement in {EARLY_STOP_PATIENCE} epochs — stopping early.")
            break


if __name__ == "__main__":
    main()
