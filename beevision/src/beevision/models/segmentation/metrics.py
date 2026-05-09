"""Segmentation eval metrics: per-class IoU + Dice, plus mean reductions.

We accumulate per-class intersection / union / (pred sum) / (target sum)
across an eval pass, then compute IoU and Dice at the end — the standard
"cumulative confusion matrix" reduction, not a batch-averaged metric.
Batch-averaged metrics lie for segmentation: a frame with very little
foreground and a frame with lots give equal weight even though one decides
the score far more.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SegMetricAccumulator:
    num_classes: int
    eps: float = 1e-6
    _inter: torch.Tensor = field(init=False)
    _union: torch.Tensor = field(init=False)
    _pred_sum: torch.Tensor = field(init=False)
    _tgt_sum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self._inter = torch.zeros(self.num_classes, dtype=torch.float64)
        self._union = torch.zeros(self.num_classes, dtype=torch.float64)
        self._pred_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self._tgt_sum = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate from logits ``(B, C, H, W)`` and target ``(B, H, W)``."""
        pred = logits.argmax(dim=1)  # (B, H, W)
        target = target.long()
        for c in range(self.num_classes):
            p = (pred == c)
            t = (target == c)
            inter = (p & t).sum().double()
            union = (p | t).sum().double()
            self._inter[c] += inter.cpu()
            self._union[c] += union.cpu()
            self._pred_sum[c] += p.sum().double().cpu()
            self._tgt_sum[c] += t.sum().double().cpu()

    def compute(self) -> dict[str, float | list[float]]:
        iou = (self._inter + self.eps) / (self._union + self.eps)
        dice = (2 * self._inter + self.eps) / (self._pred_sum + self._tgt_sum + self.eps)
        total_inter = self._inter.sum()
        total_tgt = self._tgt_sum.sum()
        pixel_acc = float(total_inter / (total_tgt + self.eps))

        out: dict[str, float | list[float]] = {
            "iou_per_class": [float(v) for v in iou.tolist()],
            "dice_per_class": [float(v) for v in dice.tolist()],
            "miou": float(iou.mean()),
            "mdice": float(dice.mean()),
            "pixel_acc": pixel_acc,
        }
        if self.num_classes == 2:
            out["iou_comb"] = float(iou[1])
            out["dice_comb"] = float(dice[1])
        return out
