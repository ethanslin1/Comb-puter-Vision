"""Cell-classifier losses.

Plain ``CrossEntropyLoss`` over 7-way logits, wrapped to return a dict with a
``loss`` and a ``ce`` component for uniform logging across heads. Class
weighting is supported but off by default — the train loader uses a
``WeightedRandomSampler`` to address ``deepbee_cls`` imbalance, so loss-side
weighting would double-correct.

Label smoothing is exposed as a config knob; with rare classes (e.g. ``egg``,
``larva``) it can help calibrate confidences but isn't on by default.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossEntropyCells(nn.Module):
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
