"""Mite classifier losses.

We train with plain ``CrossEntropyLoss`` over 2-way logits. A wrapping module
returns a dict ``{"loss", "ce"}`` — same shape as ``CEDiceLoss`` returns —
so the training loop can log loss components uniformly across heads.

Class weighting is supported but off by default. The combined varroa +
beeimage train set is roughly balanced after per-source binarization, and
the train loader uses ``WeightedRandomSampler`` for any residual skew, so
loss-side weighting would double-correct.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossEntropyMite(nn.Module):
    """``CrossEntropyLoss`` wrapper that returns a logging-friendly dict."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(label_smoothing),
        )

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        ce = self.ce(logits, target.long())
        return {"loss": ce, "ce": ce.detach()}
