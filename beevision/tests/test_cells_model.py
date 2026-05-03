"""Tests for the ResNet-50 7-class cell classifier.

Mirrors ``test_mite_model.py`` adapted to multi-class. We avoid the ImageNet
weights download by injecting a fake backbone with the same forward shape
(``(B, 3, H, W) -> (B, num_classes)``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from beevision.models.cells.losses import CrossEntropyCells
from beevision.models.cells.metrics import CellMetricAccumulator
from beevision.models.cells.resnet50 import (
    CellModelConfig,
    ResNet50Cells,
)


# ---------- Fake backbone --------------------------------------------------


class _FakeResNet50Cells(torch.nn.Module):
    """Stand-in for ``ResNet50Cells``: tiny conv stack + global pool + FC."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.cfg = CellModelConfig(num_classes=num_classes, pretrained=False)
        self.frozen = False
        self.backbone = torch.nn.Sequential()  # placeholder for parity

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def _make_model(num_classes: int = 7) -> _FakeResNet50Cells:
    return _FakeResNet50Cells(num_classes=num_classes)


# ---------- Forward shapes -------------------------------------------------


@pytest.mark.parametrize("image_size", [56, 112, 224])
def test_forward_shape_is_logits_per_sample(image_size: int) -> None:
    model = _make_model(num_classes=7)
    x = torch.randn(3, 3, image_size, image_size)
    y = model(x)
    assert y.shape == (3, 7)
    assert y.dtype == torch.float32


def test_real_resnet50_head_is_seven_way() -> None:
    """Build the real model with pretrained=False and check the FC head."""
    cfg = CellModelConfig(pretrained=False, num_classes=7)
    model = ResNet50Cells(cfg)
    fc = model.backbone.fc
    assert isinstance(fc, torch.nn.Linear)
    assert fc.out_features == 7
    assert fc.in_features == 2048


def test_freeze_backbone_excludes_only_fc_from_trainable() -> None:
    cfg = CellModelConfig(pretrained=False, freeze_backbone=True, num_classes=7)
    model = ResNet50Cells(cfg)
    trainable = {id(p) for p in model.trainable_parameters()}
    fc_params = {id(p) for p in model.backbone.fc.parameters()}
    assert trainable == fc_params
    assert all(p.requires_grad for p in model.backbone.fc.parameters())


# ---------- Losses ---------------------------------------------------------


def test_cross_entropy_returns_loss_and_ce_dict() -> None:
    crit = CrossEntropyCells()
    logits = torch.randn(4, 7, requires_grad=True)
    target = torch.tensor([0, 3, 5, 2])
    parts = crit(logits, target)
    assert "loss" in parts and "ce" in parts
    assert parts["loss"].requires_grad
    parts["loss"].backward()
    assert logits.grad is not None


def test_cross_entropy_perfect_prediction_near_zero() -> None:
    crit = CrossEntropyCells()
    # 7 samples, each strongly argmax to its own class.
    logits = torch.full((7, 7), -10.0)
    for i in range(7):
        logits[i, i] = 10.0
    target = torch.arange(7)
    loss = crit(logits, target)["loss"]
    assert loss.item() < 1e-3


# ---------- Metrics --------------------------------------------------------


def _perfect_logits(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build logits that strongly argmax to the given target."""
    n = target.shape[0]
    logits = torch.full((n, num_classes), -10.0)
    for i, t in enumerate(target.tolist()):
        logits[i, int(t)] = 10.0
    return logits


def test_metric_accumulator_perfect_predictions() -> None:
    acc = CellMetricAccumulator(num_classes=7)
    target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 3])  # all 7 classes present
    logits = _perfect_logits(target, 7)
    acc.update(logits, target)
    out = acc.compute()
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["balanced_accuracy"] == pytest.approx(1.0)
    assert out["macro_f1"] == pytest.approx(1.0)


def test_metric_accumulator_handles_absent_class_in_eval() -> None:
    """The ``other`` class is empty in train/val — shouldn't crash or pollute."""
    acc = CellMetricAccumulator(num_classes=7)
    # Use only classes 0..5 (skip 6 = "other"), perfect predictions.
    target = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])
    logits = _perfect_logits(target, 7)
    acc.update(logits, target)
    out = acc.compute()
    assert out["accuracy"] == pytest.approx(1.0)
    # Balanced accuracy averages only over present classes (0..5).
    assert out["balanced_accuracy"] == pytest.approx(1.0)
    # Macro F1 also masks to present classes.
    assert out["macro_f1"] == pytest.approx(1.0)
    # Bookkeeping: classes_present marks the absent class.
    assert out["classes_present"][6] is False
    assert all(out["classes_present"][:6])


def test_metric_accumulator_per_class_breakdown() -> None:
    acc = CellMetricAccumulator(num_classes=7)
    # 4 truth-class-0 samples; we predict 3 right and 1 as class 1.
    target = torch.tensor([0, 0, 0, 0])
    logits = torch.tensor([
        [10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # → 0 correct
        [10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # → 0 correct
        [10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # → 0 correct
        [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # → 1 wrong
    ])
    acc.update(logits, target)
    out = acc.compute()
    # Class 0 recall = 3/4 (caught 3 of 4 truths).
    assert out["recall_per_class"][0] == pytest.approx(0.75)
    # Class 0 precision = 3/3 (when we predicted 0, we were right).
    assert out["precision_per_class"][0] == pytest.approx(1.0, abs=1e-6)
    # Class 1: predicted once, never present in truths → precision 0, recall 0.
    assert out["precision_per_class"][1] == pytest.approx(0.0, abs=1e-6)
    assert out["recall_per_class"][1] == pytest.approx(0.0, abs=1e-6)


def test_metric_accumulator_imbalance_balanced_vs_raw_accuracy() -> None:
    """Predicting always-majority gets high accuracy but low balanced accuracy."""
    acc = CellMetricAccumulator(num_classes=7)
    # 9 class-0 samples + 1 class-3 sample; predict 0 always.
    target = torch.tensor([0] * 9 + [3])
    logits = torch.full((10, 7), -10.0)
    logits[:, 0] = 10.0  # always predict 0
    acc.update(logits, target)
    out = acc.compute()
    assert out["accuracy"] == pytest.approx(0.9)
    # Two classes present (0 and 3); recall_0 = 1.0, recall_3 = 0.0 → bal_acc = 0.5.
    assert out["balanced_accuracy"] == pytest.approx(0.5)


# ---------- Overfit smoke test --------------------------------------------


def test_can_overfit_one_batch() -> None:
    """Sanity: loss drops materially on a learnable signal across 7 classes."""
    torch.manual_seed(0)
    model = _make_model(num_classes=7)
    crit = CrossEntropyCells()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=3e-3)

    # 7 samples, each at a different brightness — class index = brightness rank.
    B = 7
    x = torch.zeros(B, 3, 56, 56)
    for i in range(B):
        x[i] = i / (B - 1)
    target = torch.arange(B)

    loss0 = crit(model(x), target)["loss"].item()
    for _ in range(120):
        opt.zero_grad()
        out = crit(model(x), target)
        out["loss"].backward()
        opt.step()
    loss1 = crit(model(x), target)["loss"].item()
    assert loss1 < loss0 * 0.5, f"loss did not drop enough: {loss0:.3f} -> {loss1:.3f}"


# ---------- Integration with CellClassificationDataset --------------------


def _write_cells_parquet(tmp_path):
    interim = tmp_path / "interim"
    rows = []
    # 6 cells across 6 of the 7 classes (skip "other"; mirrors train/val reality).
    classes = ["egg", "larva", "capped_brood", "pollen", "nectar", "honey"]
    for i, label in enumerate(classes):
        img_rel = f"images/deepbee_cls/train/c{i}.png"
        (interim / img_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (56, 56), color=(30 + 30 * i, 60, 100)).save(interim / img_rel)
        rows.append({
            "id": f"deepbee_cls:train:c{i}",
            "source": "deepbee_cls",
            "split": "train",
            "image_path": img_rel,
            "label": label,
            "meta": {"classes": classes},
        })
    pq = interim / "deepbee_cls.parquet"
    pd.DataFrame(rows).to_parquet(pq, index=False, engine="pyarrow")
    return pq, interim


def test_model_consumes_dataset_batch(tmp_path) -> None:
    from beevision.data.augment import classification_transforms
    from beevision.data.datasets import CellClassificationDataset

    pq, interim = _write_cells_parquet(tmp_path)
    tf = classification_transforms(image_size=56, train=False)
    ds = CellClassificationDataset(pq, interim, split="train", transform=tf)

    batch = [ds[i] for i in range(len(ds))]
    images = torch.from_numpy(np.stack([b["image"] for b in batch]))
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    assert images.shape == (6, 3, 56, 56) and images.dtype == torch.float32
    assert targets.shape == (6,) and targets.dtype == torch.int64
    # The 6 unique labels should map to 6 distinct class indices.
    assert len(set(targets.tolist())) == 6

    model = _make_model(num_classes=7)
    logits = model(images)
    assert logits.shape == (6, 7)

    crit = CrossEntropyCells()
    parts = crit(logits, targets)
    assert torch.isfinite(parts["loss"])
