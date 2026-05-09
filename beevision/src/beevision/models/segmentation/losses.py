"""Segmentation losses.

We train with ``CrossEntropy + SoftDice`` — CE gives a per-pixel gradient on
confidence, Dice gives an overlap objective that's robust to the large
background class in hive frames (comb typically covers 30–70% of pixels, so
class imbalance is real but not catastrophic; still, Dice helps).

Both losses operate on logits of shape ``(B, C, H, W)`` and integer targets
of shape ``(B, H, W)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice over softmax probabilities.

    Dice per class = (2·|P∩T|) / (|P| + |T| + eps). Loss = 1 − mean_over_classes.
    Background (class 0) is included by default; set ``ignore_background=True``
    to compute Dice only over foreground classes (common for medical seg).
    """

    def __init__(self, eps: float = 1e-6, ignore_background: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  
        onehot = F.one_hot(target.long(), num_classes=num_classes) 
        onehot = onehot.permute(0, 3, 1, 2).float() 

        dims = (0, 2, 3)  
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)  # (C,)

        if self.ignore_background and num_classes > 1:
            dice = dice[1:]
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """Weighted sum of ``CrossEntropyLoss`` and ``SoftDiceLoss``."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
        ignore_background_in_dice: bool = False,
    ) -> None:
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = SoftDiceLoss(ignore_background=ignore_background_in_dice)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        ce = self.ce(logits, target.long())
        dice = self.dice(logits, target)
        total = self.ce_weight * ce + self.dice_weight * dice
        return {"loss": total, "ce": ce.detach(), "dice": dice.detach()}
