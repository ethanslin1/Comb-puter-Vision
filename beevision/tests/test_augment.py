"""Unit tests for beevision.data.augment."""
from __future__ import annotations

import numpy as np
import pytest

from beevision.data.augment import (
    apply,
    classification_transforms,
    seed_pipeline,
    segmentation_transforms,
)


# ---------- Output shape / dtype contract ---------------------------------


def test_classification_eval_returns_chw_float_normalized() -> None:
    img = np.random.default_rng(0).integers(0, 256, size=(300, 200, 3), dtype=np.uint8)
    tf = classification_transforms(image_size=224, train=False)
    out = apply(tf, img)
    assert out["image"].shape == (3, 224, 224)
    assert out["image"].dtype == np.float32
    assert 0.0 <= float(out["image"].min()) <= float(out["image"].max()) <= 1.0


def test_classification_train_same_shape_as_eval() -> None:
    img = np.random.default_rng(0).integers(0, 256, size=(300, 200, 3), dtype=np.uint8)
    tf = classification_transforms(image_size=224, train=True)
    out = apply(seed_pipeline(tf, 0), img)
    assert out["image"].shape == (3, 224, 224)
    assert "mask" not in out


def test_segmentation_eval_preserves_label_set_and_shape() -> None:
    img = np.random.default_rng(0).integers(0, 256, size=(300, 200, 3), dtype=np.uint8)
    mask = np.zeros((300, 200), dtype=np.uint8)
    mask[:, 100:] = 1
    tf = segmentation_transforms(image_size=256, train=False)
    out = apply(tf, img, mask)
    assert out["image"].shape == (3, 256, 256)
    assert out["mask"].shape == (256, 256)
    assert out["mask"].dtype == np.int64
    # Eval uses reflect pad + long-edge resize; labels never interpolate
    # to values outside {0, 1}.
    unique = set(np.unique(out["mask"]).tolist())
    assert unique <= {0, 1}, f"eval mask had extra values: {unique}"


def test_segmentation_train_keeps_mask_binary() -> None:
    img = np.random.default_rng(1).integers(0, 256, size=(512, 800, 3), dtype=np.uint8)
    mask = np.zeros((512, 800), dtype=np.uint8)
    mask[128:384, 200:600] = 1
    tf = segmentation_transforms(image_size=256, train=True)
    out = apply(seed_pipeline(tf, 0), img, mask)
    unique = set(np.unique(out["mask"]).tolist())
    assert unique <= {0, 1}, f"train mask had extra values: {unique}"
    assert out["image"].shape == (3, 256, 256)
    assert out["mask"].shape == (256, 256)


# ---------- Determinism ----------------------------------------------------


def test_train_pipeline_is_seeded_reproducibly() -> None:
    img = np.random.default_rng(42).integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    tf = classification_transforms(image_size=224, train=True)
    a = apply(seed_pipeline(tf, 7), img)
    b = apply(seed_pipeline(tf, 7), img)
    # With identical seeds + identical transform, output should match.
    assert np.array_equal(a["image"], b["image"])


def test_eval_pipeline_is_deterministic_without_seeding() -> None:
    img = np.random.default_rng(0).integers(0, 256, size=(200, 250, 3), dtype=np.uint8)
    tf = classification_transforms(image_size=224, train=False)
    a = apply(tf, img)
    b = apply(tf, img)
    assert np.array_equal(a["image"], b["image"])


@pytest.mark.parametrize("size", [128, 224, 384])
def test_classification_eval_respects_target_size(size: int) -> None:
    img = np.random.default_rng(0).integers(0, 256, size=(123, 456, 3), dtype=np.uint8)
    tf = classification_transforms(image_size=size, train=False)
    out = apply(tf, img)
    assert out["image"].shape == (3, size, size)
