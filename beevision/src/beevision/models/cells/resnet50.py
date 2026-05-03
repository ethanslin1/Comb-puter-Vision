"""ResNet-50 7-class cell classifier.

Architecture
------------
- **Backbone**: ``torchvision.models.resnet50`` with ImageNet weights by default,
  fine-tuned end-to-end. With ~46k labeled cells in train, freezing the backbone
  is unnecessarily restrictive; ``freeze_backbone`` is a config flag for ablations.
- **Head**: stock 1000-way FC replaced with ``Linear(2048, 7)``. Outputs are
  logits ``(B, 7)`` over ``CELL_CLASSES`` from ``data/datasets.py``:
  ``(egg, larva, capped_brood, pollen, nectar, honey, other)``.

Note on the ``other`` class: ``deepbee_cls`` only contains ``other``-labeled
cells in its test split (per the README's per-source notes). The model still
emits 7 logits — at train/val time the 7th class simply has no positive
examples, so the loss never punishes predictions there. Macro metrics that
iterate over all 7 classes will treat the absent class as F1=0 in train/val,
which is honest reporting.

Input shape is ``(B, 3, 224, 224)``; matches torchvision ImageNet preprocessing
and the ``crop=224`` size produced by the classification augment pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

LOG = logging.getLogger("models.cells")

RESNET50_FEATURE_DIM = 2048


# ---------- Config ---------------------------------------------------------


@dataclass
class CellModelConfig:
    num_classes: int = 7
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout: float = 0.0


# ---------- Model ----------------------------------------------------------


class ResNet50Cells(nn.Module):
    """torchvision ResNet-50 with a 7-class classification head."""

    def __init__(self, cfg: CellModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CellModelConfig()
        if self.cfg.backbone != "resnet50":
            raise ValueError(
                f"only resnet50 is supported here; got {self.cfg.backbone!r}"
            )

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

    def train(self, mode: bool = True) -> "ResNet50Cells":
        super().train(mode)
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


def build_model(cfg: dict[str, Any] | CellModelConfig | None = None) -> ResNet50Cells:
    """Build a ``ResNet50Cells`` from a dict config or ``CellModelConfig``."""
    if cfg is None:
        return ResNet50Cells()
    if isinstance(cfg, CellModelConfig):
        return ResNet50Cells(cfg)
    model_cfg = CellModelConfig(
        num_classes=int(cfg.get("num_classes", 7)),
        backbone=str(cfg.get("backbone", "resnet50")),
        pretrained=bool(cfg.get("pretrained", True)),
        freeze_backbone=bool(cfg.get("freeze_backbone", False)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    return ResNet50Cells(model_cfg)
