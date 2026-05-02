"""ResNet-50 binary mite classifier.

Architecture
------------
- **Backbone**: ``torchvision.models.resnet50`` with ImageNet weights by default.
  Unlike the segmentation encoder we do **not** freeze the backbone — the mite
  head trains on ~11k labeled bee crops (varroa + beeimage combined), enough
  to fine-tune the whole network without immediate overfit. Freeze flag is
  available for ablations.
- **Head**: the stock 1000-way FC is replaced with a 2-way ``Linear(2048, 2)``.
  Outputs are logits ``(B, 2)`` — apply softmax for probabilities, argmax for
  class indices. Class index 0 = ``no_mite``, 1 = ``mite``, matching
  ``MITE_CLASSES`` in ``data/datasets.py``.

Input shape is ``(B, 3, 224, 224)`` — square 224 matches torchvision's
ImageNet preprocessing and the ``crop=224`` size already produced by the
classification augment pipeline. Larger inputs work but waste compute since
the encoder only saw 224 in pretraining.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

LOG = logging.getLogger("models.mite")

# Stock torchvision FC width for resnet50.
RESNET50_FEATURE_DIM = 2048


# ---------- Config ---------------------------------------------------------


@dataclass
class MiteModelConfig:
    num_classes: int = 2
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout: float = 0.0  # optional pre-FC dropout; 0 = no-op


# ---------- Model ----------------------------------------------------------


class ResNet50Mite(nn.Module):
    """torchvision ResNet-50 with a binary classification head.

    The original 1000-way FC at ``model.fc`` is swapped for an optional
    Dropout + Linear(2048, num_classes). All other layers retain their
    pretrained weights.
    """

    def __init__(self, cfg: MiteModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MiteModelConfig()
        if self.cfg.backbone != "resnet50":
            raise ValueError(
                f"only resnet50 is supported here; got {self.cfg.backbone!r}"
            )

        # Lazy import so tests that pass a fake backbone don't pay the
        # torchvision weight download.
        from torchvision import models

        weights = models.ResNet50_Weights.DEFAULT if self.cfg.pretrained else None
        backbone = models.resnet50(weights=weights)

        if self.cfg.dropout > 0:
            backbone.fc = nn.Sequential(
                nn.Dropout(p=self.cfg.dropout),
                nn.Linear(RESNET50_FEATURE_DIM, self.cfg.num_classes),
            )
        else:
            backbone.fc = nn.Linear(RESNET50_FEATURE_DIM, self.cfg.num_classes)

        self.backbone = backbone
        self.frozen = self.cfg.freeze_backbone
        if self.frozen:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    p.requires_grad = False

    def train(self, mode: bool = True) -> "ResNet50Mite":
        super().train(mode)
        # Keep BatchNorm stats frozen when the backbone is frozen — otherwise
        # BN running stats drift even with requires_grad=False, hurting eval.
        if self.frozen:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# ---------- Factory --------------------------------------------------------


def build_model(cfg: dict[str, Any] | MiteModelConfig | None = None) -> ResNet50Mite:
    """Build a ``ResNet50Mite`` from a dict config or ``MiteModelConfig``."""
    if cfg is None:
        return ResNet50Mite()
    if isinstance(cfg, MiteModelConfig):
        return ResNet50Mite(cfg)
    model_cfg = MiteModelConfig(
        num_classes=int(cfg.get("num_classes", 2)),
        backbone=str(cfg.get("backbone", "resnet50")),
        pretrained=bool(cfg.get("pretrained", True)),
        freeze_backbone=bool(cfg.get("freeze_backbone", False)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    return ResNet50Mite(model_cfg)
