"""Tests for the ResNet-50 mite classifier.

We avoid the ~100MB ImageNet weights download in unit tests by injecting a
fake backbone module that mirrors ``ResNet50Mite``'s expected interface
(``forward(x: (B, 3, H, W)) -> (B, num_classes)``). That keeps tests
offline and fast while still exercising the loss, metrics, and dataset
integration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from beevision.models.mite.losses import CrossEntropyMite
from beevision.models.mite.metrics import (
    ClassificationMetricAccumulator,
    _binary_auroc,
)
from beevision.models.mite.resnet50 import (
    MiteModelConfig,
    ResNet50Mite,
)


# ---------- Fake backbone --------------------------------------------------


class _FakeResNet50Mite(torch.nn.Module):
    """Stand-in for ``ResNet50Mite``: tiny conv stack + global pool + FC.

    Same forward signature as the real model — ``(B, 3, H, W) -> (B, num_classes)``
    — and exposes ``.backbone.fc`` like torchvision's ResNet so any code that
    inspects the head still works.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.cfg = MiteModelConfig(num_classes=num_classes, pretrained=False)
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


def _make_model(num_classes: int = 2) -> _FakeResNet50Mite:
    return _FakeResNet50Mite(num_classes=num_classes)


# ---------- Forward shapes -------------------------------------------------


@pytest.mark.parametrize("image_size", [56, 112, 224])
def test_forward_shape_is_logits_per_sample(image_size: int) -> None:
    model = _make_model(num_classes=2)
    x = torch.randn(3, 3, image_size, image_size)
    y = model(x)
    assert y.shape == (3, 2)
    assert y.dtype == torch.float32


def test_real_resnet50_head_replaced_to_num_classes() -> None:
    """Build the real model with pretrained=False (no download) and check the FC head."""
    cfg = MiteModelConfig(pretrained=False, num_classes=2)
    model = ResNet50Mite(cfg)
    fc = model.backbone.fc
    assert isinstance(fc, torch.nn.Linear)
    assert fc.out_features == 2
    assert fc.in_features == 2048


def test_freeze_backbone_excludes_only_fc_from_trainable() -> None:
    cfg = MiteModelConfig(pretrained=False, freeze_backbone=True, num_classes=2)
    model = ResNet50Mite(cfg)
    trainable = {id(p) for p in model.trainable_parameters()}
    fc_params = {id(p) for p in model.backbone.fc.parameters()}
    # Every trainable param must be from the FC head when frozen.
    assert trainable == fc_params
    # And the FC must be trainable in full.
    assert all(p.requires_grad for p in model.backbone.fc.parameters())


# ---------- Losses ---------------------------------------------------------


def test_cross_entropy_returns_loss_and_ce_dict() -> None:
    crit = CrossEntropyMite()
    logits = torch.randn(4, 2, requires_grad=True)
    target = torch.tensor([0, 1, 1, 0])
    parts = crit(logits, target)
    assert "loss" in parts and "ce" in parts
    assert parts["loss"].requires_grad
    parts["loss"].backward()
    assert logits.grad is not None


def test_cross_entropy_perfect_prediction_near_zero() -> None:
    crit = CrossEntropyMite()
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    target = torch.tensor([0, 1])
    loss = crit(logits, target)["loss"]
    assert loss.item() < 1e-3


# ---------- Metrics --------------------------------------------------------


def test_metric_accumulator_perfect_predictions() -> None:
    acc = ClassificationMetricAccumulator(num_classes=2)
    target = torch.tensor([0, 0, 1, 1])
    # Logits strongly argmax to target.
    logits = torch.tensor([[10.0, -10.0], [10.0, -10.0], [-10.0, 10.0], [-10.0, 10.0]])
    acc.update(logits, target)
    out = acc.compute()
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["balanced_accuracy"] == pytest.approx(1.0)
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["confusion"] == [[2, 0], [0, 2]]


def test_metric_accumulator_all_wrong() -> None:
    acc = ClassificationMetricAccumulator(num_classes=2)
    target = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[-10.0, 10.0], [-10.0, 10.0], [10.0, -10.0], [10.0, -10.0]])
    acc.update(logits, target)
    out = acc.compute()
    assert out["accuracy"] == pytest.approx(0.0)
    assert out["balanced_accuracy"] == pytest.approx(0.0)
    assert out["confusion"] == [[0, 2], [2, 0]]


def test_metric_accumulator_balanced_accuracy_under_skew() -> None:
    """Predict majority class everywhere when one class dominates.

    With 9 negatives and 1 positive, predicting all negative gives accuracy
    = 0.9 but balanced accuracy = 0.5 (recall_neg=1.0, recall_pos=0.0). This
    is exactly why we use balanced accuracy as the best-metric.
    """
    acc = ClassificationMetricAccumulator(num_classes=2)
    target = torch.tensor([0] * 9 + [1])
    logits = torch.full((10, 2), -10.0)
    logits[:, 0] = 10.0  # always predict class 0
    acc.update(logits, target)
    out = acc.compute()
    assert out["accuracy"] == pytest.approx(0.9)
    assert out["balanced_accuracy"] == pytest.approx(0.5)


def test_binary_auroc_perfect_separator() -> None:
    # All positives outrank all negatives → AUROC = 1.
    auc = _binary_auroc(scores=[0.1, 0.2, 0.8, 0.9], targets=[0, 0, 1, 1])
    assert auc == pytest.approx(1.0)


def test_binary_auroc_random_is_one_half_with_ties() -> None:
    # All scores tied → mean rank for everyone → AUROC = 0.5.
    auc = _binary_auroc(scores=[0.5, 0.5, 0.5, 0.5], targets=[0, 0, 1, 1])
    assert auc == pytest.approx(0.5)


def test_binary_auroc_handles_single_class() -> None:
    # No positive samples → AUROC is undefined; we return NaN.
    auc = _binary_auroc(scores=[0.1, 0.2, 0.3], targets=[0, 0, 0])
    assert np.isnan(auc)


# ---------- Overfit smoke test --------------------------------------------


def test_can_overfit_one_batch() -> None:
    """Sanity: loss drops materially on a learnable signal.

    Class 1 = bright, class 0 = dark. A tiny conv stack should learn this in
    a few dozen steps; if it doesn't, the wiring (logits → loss → grad) is
    broken.
    """
    torch.manual_seed(0)
    model = _make_model(num_classes=2)
    crit = CrossEntropyMite()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=3e-3)

    B = 8
    x = torch.zeros(B, 3, 56, 56)
    x[B // 2:] = 1.0  # second half is bright
    target = torch.tensor([0] * (B // 2) + [1] * (B // 2))

    loss0 = crit(model(x), target)["loss"].item()
    for _ in range(80):
        opt.zero_grad()
        out = crit(model(x), target)
        out["loss"].backward()
        opt.step()
    loss1 = crit(model(x), target)["loss"].item()
    assert loss1 < loss0 * 0.5, f"loss did not drop enough: {loss0:.3f} -> {loss1:.3f}"


# ---------- Integration with MiteClassificationDataset --------------------


def _write_mite_parquet(tmp_path, source: str = "varroa", split: str = "train"):
    interim = tmp_path / "interim"
    rows = []
    for i in range(4):
        img_rel = f"images/{source}/{split}/c{i}.png"
        (interim / img_rel).parent.mkdir(parents=True, exist_ok=True)
        # Two no_mite then two mite — alternating brightness so they're learnable.
        color = (50, 60, 70) if i < 2 else (200, 210, 220)
        Image.new("RGB", (56, 56), color=color).save(interim / img_rel)
        rows.append({
            "id": f"{source}:{split}:c{i}",
            "source": source,
            "split": split,
            "image_path": img_rel,
            "label": "no_mite" if i < 2 else "mite",
            "meta": {"classes": ["no_mite", "mite"]},
        })
    pq = interim / f"{source}.parquet"
    pd.DataFrame(rows).to_parquet(pq, index=False, engine="pyarrow")
    return pq, interim


def test_model_consumes_dataset_batch(tmp_path) -> None:
    from beevision.data.augment import classification_transforms
    from beevision.data.datasets import MiteClassificationDataset

    pq, interim = _write_mite_parquet(tmp_path)
    tf = classification_transforms(image_size=56, train=False)
    ds = MiteClassificationDataset(pq, interim, split="train", transform=tf)

    batch = [ds[i] for i in range(len(ds))]
    images = torch.from_numpy(np.stack([b["image"] for b in batch]))
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    assert images.shape == (4, 3, 56, 56) and images.dtype == torch.float32
    assert targets.shape == (4,) and targets.dtype == torch.int64

    model = _make_model(num_classes=2)
    logits = model(images)
    assert logits.shape == (4, 2)

    crit = CrossEntropyMite()
    parts = crit(logits, targets)
    assert torch.isfinite(parts["loss"])
