"""Unit tests for beevision.data.ingest_deepbee_cls helpers."""
from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from beevision.data.ingest_deepbee_cls import (
    carve_val,
    center_crop,
    normalize_label,
    reconcile,
)


LABEL_MAP = {
    "eggs": "egg",
    "egg": "egg",
    "larves": "larva",
    "larva": "larva",
    "capped": "capped_brood",
    "pollen": "pollen",
    "nectar": "nectar",
    "honey": "honey",
    "other": "other",
}


# ---------- normalize_label ------------------------------------------------


def test_normalize_label_pt_to_en() -> None:
    assert normalize_label("eggs", LABEL_MAP) == "egg"
    assert normalize_label("larves", LABEL_MAP) == "larva"
    assert normalize_label("capped", LABEL_MAP) == "capped_brood"


def test_normalize_label_case_and_whitespace() -> None:
    assert normalize_label("  Larves  ", LABEL_MAP) == "larva"
    assert normalize_label("NECTAR", LABEL_MAP) == "nectar"


def test_normalize_label_unknown_returns_none() -> None:
    assert normalize_label("dontcare", LABEL_MAP) is None
    assert normalize_label("", LABEL_MAP) is None
    assert normalize_label(None, LABEL_MAP) is None
    assert normalize_label("garbage", LABEL_MAP) is None


# ---------- reconcile ------------------------------------------------------


def _df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cell_id", "x", "y", "label", "image"])


def test_reconcile_drops_missing_test_images_and_overlapping_train() -> None:
    train = _df([
        (1, 10, 10, "honey", "A.JPG"),
        (2, 20, 20, "honey", "B.JPG"),   # will be in test → drop from train
        (3, 30, 30, "pollen", "C.JPG"),
    ])
    test = _df([
        (100, 40, 40, "nectar", "B.JPG"),  # on disk → keep
        (101, 50, 50, "other", "MISSING.JPG"),  # off disk → drop
    ])
    on_disk = {"A.JPG", "B.JPG", "C.JPG"}
    train_out, test_out, stats = reconcile(train, test, on_disk)
    assert set(test_out["image"]) == {"B.JPG"}
    assert set(train_out["image"]) == {"A.JPG", "C.JPG"}
    assert stats["test_rows_missing_image"] == 1
    assert stats["train_rows_dropped_test_overlap"] == 1
    assert stats["train_rows_missing_image"] == 0


def test_reconcile_drops_train_rows_whose_image_not_on_disk() -> None:
    train = _df([
        (1, 10, 10, "honey", "ON_DISK.JPG"),
        (2, 20, 20, "honey", "OFF_DISK.JPG"),
    ])
    test = _df([])
    train_out, test_out, stats = reconcile(train, test, {"ON_DISK.JPG"})
    assert list(train_out["image"]) == ["ON_DISK.JPG"]
    assert stats["train_rows_missing_image"] == 1
    assert len(test_out) == 0


# ---------- carve_val ------------------------------------------------------


def test_carve_val_deterministic_and_stratified() -> None:
    df = pd.DataFrame(
        {
            "cell_id": range(200),
            "x": 0,
            "y": 0,
            "label": (["honey"] * 140) + (["egg"] * 60),
            "image": "X.JPG",
        }
    )
    out_a = carve_val(df, val_fraction=0.15, seed=1337)
    out_b = carve_val(df, val_fraction=0.15, seed=1337)
    assert list(out_a["split"]) == list(out_b["split"])

    counts = out_a.groupby("split").size()
    # 15% of 200 = 30; allow ±2 for stratification rounding.
    assert abs(counts["val"] - 30) <= 2
    # Val class mix should mirror global (70% honey, 30% egg) within ±10 pp.
    val = out_a[out_a["split"] == "val"]
    honey_frac = (val["label"] == "honey").mean()
    assert abs(honey_frac - 0.70) < 0.10


def test_carve_val_rejects_bad_fraction() -> None:
    df = pd.DataFrame({"cell_id": [1, 2, 3, 4], "label": ["a", "a", "b", "b"]})
    with pytest.raises(ValueError):
        carve_val(df, val_fraction=0.0, seed=0)
    with pytest.raises(ValueError):
        carve_val(df, val_fraction=1.0, seed=0)


# ---------- center_crop ----------------------------------------------------


def test_center_crop_interior_exact_center() -> None:
    img = Image.new("RGB", (1000, 800), color=(0, 0, 0))
    # Paint a marker at (500, 400) so we can check it's centered.
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            img.putpixel((500 + dx, 400 + dy), (255, 0, 0))
    crop = center_crop(img, 500, 400, 224)
    assert crop.size == (224, 224)
    assert crop.getpixel((112, 112)) == (255, 0, 0)


def test_center_crop_edge_shifts_box_in_bounds() -> None:
    img = Image.new("RGB", (1000, 800), color=(0, 0, 0))
    crop = center_crop(img, x=10, y=10, side=224)
    # Always side×side, always within image.
    assert crop.size == (224, 224)


def test_center_crop_corner() -> None:
    img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    crop = center_crop(img, x=299, y=299, side=100)
    assert crop.size == (100, 100)


def test_center_crop_upscales_small_frames() -> None:
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    # side > min(W,H) → upscale then center.
    crop = center_crop(img, x=50, y=50, side=224)
    assert crop.size == (224, 224)
