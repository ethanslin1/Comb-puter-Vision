"""Unit tests for beevision.data.ingest_deepbee_seg helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from beevision.data.ingest_deepbee_seg import (
    decode_mask,
    resize_short_edge,
    stratify_by_fg_tercile,
)


# ---------- decode_mask ----------------------------------------------------


def _gray_rgb(h: int, w: int, value: int) -> np.ndarray:
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return arr


def test_decode_mask_binary_grayscale_rgb() -> None:
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:, 5:] = 255  # right half is foreground
    mask, stats = decode_mask(arr, threshold_gray=128, channel_consistency_tol=8)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert mask[:, :5].sum() == 0
    assert mask[:, 5:].sum() == 5 * 10  # 50 foreground pixels
    assert stats["fg_frac"] == 0.5
    assert stats["ambiguous_frac"] == 0.0
    assert stats["p99_channel_diff"] == 0


def test_decode_mask_accepts_jpeg_edge_artifacts() -> None:
    """A few boundary pixels with R≠B must not abort — only p99 matters."""
    arr = _gray_rgb(100, 100, 0).copy()
    arr[:, 50:] = 255
    # Inject 10 isolated artifact pixels with huge channel disagreement.
    idx = np.random.default_rng(0).integers(0, 100, size=(10, 2))
    for r, c in idx:
        arr[r, c] = [10, 200, 10]
    mask, stats = decode_mask(arr, threshold_gray=128, channel_consistency_tol=8)
    # p99 stays small since only 0.1% of pixels are bad.
    assert stats["p99_channel_diff"] <= 8
    assert mask.shape == (100, 100)


def test_decode_mask_aborts_on_true_color() -> None:
    """If channels disagree pervasively, abort (spec: fail loudly)."""
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[..., 0] = 200  # wall-to-wall red
    with pytest.raises(ValueError):
        decode_mask(arr, threshold_gray=128, channel_consistency_tol=8)


def test_decode_mask_reports_ambiguous_fraction() -> None:
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:5, :, :] = 64     # ambiguous mid-gray
    arr[5:, :, :] = 255
    _, stats = decode_mask(arr, threshold_gray=128, channel_consistency_tol=8)
    assert stats["ambiguous_frac"] == pytest.approx(0.5)


def test_decode_mask_accepts_2d_grayscale_input() -> None:
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[:, 4:] = 255
    mask, stats = decode_mask(arr, threshold_gray=128, channel_consistency_tol=8)
    assert mask.shape == (8, 8)
    assert stats["fg_frac"] == 0.5


# ---------- resize_short_edge ----------------------------------------------


def test_resize_short_edge_image_uses_lanczos() -> None:
    img = Image.new("RGB", (800, 600), color=(128, 128, 128))
    out = resize_short_edge(img, 300, is_mask=False)
    w, h = out.size
    assert min(w, h) == 300
    assert abs((w / h) - (800 / 600)) < 0.01


def test_resize_short_edge_mask_uses_nearest_preserves_labels() -> None:
    # Mask with class values {0, 1} must never be interpolated to 127 etc.
    arr = np.zeros((100, 120), dtype=np.uint8)
    arr[:, 60:] = 1
    img = Image.fromarray(arr, mode="L")
    out = resize_short_edge(img, 50, is_mask=True)
    unique_vals = set(np.asarray(out).flatten().tolist())
    assert unique_vals <= {0, 1}, f"got {unique_vals}"


def test_resize_short_edge_no_op_when_already_sized() -> None:
    img = Image.new("RGB", (400, 300), color=0)
    out = resize_short_edge(img, 300, is_mask=False)
    assert out.size == (400, 300)


# ---------- stratify_by_fg_tercile -----------------------------------------


def test_stratify_is_deterministic_and_exhausts_input() -> None:
    df = pd.DataFrame(
        {
            "image_name": [f"{i}.JPG" for i in range(60)],
            "fg_frac": np.linspace(0.1, 0.9, 60),
        }
    )
    a = stratify_by_fg_tercile(df, val_fraction=0.15, test_fraction=0.15, seed=1337)
    b = stratify_by_fg_tercile(df, val_fraction=0.15, test_fraction=0.15, seed=1337)
    assert list(a["split"]) == list(b["split"])
    assert set(a["split"]) == {"train", "val", "test"}
    assert len(a) == len(df)
    counts = a.groupby("split").size()
    # sklearn stratified round-halves may nudge by ±1.
    assert abs(counts["test"] - 9) <= 1
    assert abs(counts["val"] - 8) <= 1
    assert counts.sum() == 60
