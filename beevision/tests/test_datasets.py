"""Tests for the torch-facing datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from beevision.data.datasets import (
    CELL_CLASSES,
    CELL_LABEL_TO_IDX,
    MITE_CLASSES,
    CellClassificationDataset,
    FrameSegmentationDataset,
    MiteClassificationDataset,
    build_mite_train_sampler,
)


def _make_png(path: Path, size=(16, 16), color=(120, 130, 140)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _make_mask(path: Path, size=(16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[:, size[0] // 2:] = 1
    Image.fromarray(arr, mode="L").save(path)


# ---------- Classification -------------------------------------------------


@pytest.fixture
def cell_parquet(tmp_path: Path) -> tuple[Path, Path]:
    interim = tmp_path / "interim"
    rows = []
    for i, label in enumerate(["egg", "larva", "honey", "pollen"]):
        rel = f"images/deepbee_cls/train/img{i}.png"
        _make_png(interim / rel)
        rows.append({
            "id": f"deepbee_cls:img{i}",
            "source": "deepbee_cls",
            "split": "train",
            "label": label,
            "image_path": rel,
            "meta": {"x": i},
        })
    # Add a val row too so split filtering is exercised.
    rel = "images/deepbee_cls/val/v0.png"
    _make_png(interim / rel)
    rows.append({
        "id": "deepbee_cls:v0",
        "source": "deepbee_cls",
        "split": "val",
        "label": "honey",
        "image_path": rel,
        "meta": {"x": 99},
    })
    pq = interim / "deepbee_cls.parquet"
    pd.DataFrame(rows).to_parquet(pq, index=False, engine="pyarrow")
    return pq, interim


def test_cell_dataset_returns_image_and_class_index(cell_parquet) -> None:
    pq, interim = cell_parquet
    ds = CellClassificationDataset(pq, interim, split="train")
    assert len(ds) == 4
    sample = ds[0]
    assert "image" in sample and "target" in sample
    assert sample["target"] in range(len(CELL_CLASSES))
    img = sample["image"]
    assert img.ndim == 3 and img.shape[-1] == 3  # HWC RGB (no transform)


def test_cell_dataset_filters_split(cell_parquet) -> None:
    pq, interim = cell_parquet
    ds_val = CellClassificationDataset(pq, interim, split="val")
    assert len(ds_val) == 1
    assert ds_val[0]["target"] == CELL_LABEL_TO_IDX["honey"]


def test_cell_dataset_with_transform_returns_chw_float(cell_parquet) -> None:
    from beevision.data.augment import classification_transforms
    pq, interim = cell_parquet
    tf = classification_transforms(image_size=32, train=False)
    ds = CellClassificationDataset(pq, interim, split="train", transform=tf)
    sample = ds[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["image"].dtype == np.float32


# ---------- Mite + sampler -------------------------------------------------


@pytest.fixture
def mite_parquet(tmp_path: Path) -> tuple[Path, Path]:
    interim = tmp_path / "interim"
    rows = []
    # 8 no_mite + 2 mite → imbalanced on purpose
    for i in range(8):
        rel = f"images/beeimage/train/nm{i}.png"
        _make_png(interim / rel)
        rows.append({
            "id": f"beeimage:nm{i}", "source": "beeimage", "split": "train",
            "label": "no_mite", "image_path": rel, "meta": {"x": i},
        })
    for i in range(2):
        rel = f"images/beeimage/train/m{i}.png"
        _make_png(interim / rel)
        rows.append({
            "id": f"beeimage:m{i}", "source": "beeimage", "split": "train",
            "label": "mite", "image_path": rel, "meta": {"x": i},
        })
    pq = interim / "beeimage.parquet"
    pd.DataFrame(rows).to_parquet(pq, index=False, engine="pyarrow")
    return pq, interim


def test_mite_train_sampler_inverse_frequency(mite_parquet) -> None:
    pq, interim = mite_parquet
    ds = MiteClassificationDataset(pq, interim, split="train")
    info = build_mite_train_sampler(ds)
    # 8 no_mite (idx 0), 2 mite (idx 1). inverse-freq: weight[mite] > weight[no_mite].
    assert info["class_weights"][1] > info["class_weights"][0]
    # Per-sample vector aligns with labels.
    assert info["sample_weights"].shape == (10,)
    # First 8 were no_mite, last 2 mite in insertion order → sorted by id keeps that.
    # Just check ratio ≈ 4 since counts are 8 vs 2.
    ratio = info["class_weights"][1] / info["class_weights"][0]
    assert abs(ratio - 4.0) < 1e-9


# ---------- Segmentation ---------------------------------------------------


@pytest.fixture
def seg_parquet(tmp_path: Path) -> tuple[Path, Path]:
    interim = tmp_path / "interim"
    rows = []
    for i in range(3):
        img_rel = f"images/deepbee_seg/train/f{i}.png"
        mask_rel = f"masks/deepbee_seg/train/f{i}.png"
        _make_png(interim / img_rel)
        _make_mask(interim / mask_rel)
        rows.append({
            "id": f"deepbee_seg:f{i}", "source": "deepbee_seg", "split": "train",
            "image_path": img_rel, "mask_path": mask_rel,
            "meta": {"classes": ["background", "comb"]},
        })
    pq = interim / "deepbee_seg.parquet"
    pd.DataFrame(rows).to_parquet(pq, index=False, engine="pyarrow")
    return pq, interim


def test_segmentation_dataset_returns_image_and_mask(seg_parquet) -> None:
    pq, interim = seg_parquet
    ds = FrameSegmentationDataset(pq, interim, split="train")
    assert len(ds) == 3
    sample = ds[0]
    assert sample["image"].ndim == 3 and sample["image"].shape[-1] == 3
    assert sample["mask"].ndim == 2
    assert set(np.unique(sample["mask"]).tolist()) <= {0, 1}


def test_segmentation_dataset_with_transform(seg_parquet) -> None:
    from beevision.data.augment import segmentation_transforms
    pq, interim = seg_parquet
    tf = segmentation_transforms(image_size=32, train=False)
    ds = FrameSegmentationDataset(pq, interim, split="train", transform=tf)
    sample = ds[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)
    assert sample["mask"].dtype == np.int64
