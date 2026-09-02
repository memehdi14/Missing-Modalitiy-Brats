"""
Dice loss and the combined Dice + BCE loss used for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice, computed per channel (WT/TC/ET) then averaged. Takes raw logits."""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        dims = (0, 2, 3, 4)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice_per_channel = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_per_channel.mean()


class DiceBCELoss(nn.Module):
    """Dice + BCE combo. Trains more stably than Dice alone."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets)