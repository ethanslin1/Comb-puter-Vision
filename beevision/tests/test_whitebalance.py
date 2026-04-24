"""Unit tests for beevision.data.whitebalance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beevision.data.whitebalance import (
    channel_means,
    gray_world,
    save_whitebalance_debug_grid,
)


# ---------- Core algorithm -------------------------------------------------


def test_gray_world_equalizes_channel_means() -> None:
    """Canonical property: after gray-world, all three channel means are
    equal (up to quantization noise)."""
    rng = np.random.default_rng(0)
    base = rng.integers(80, 180, size=(64, 64, 3), dtype=np.uint8).astype(np.float64)
    # Introduce a strong warm cast: R up, B down.
    base[..., 0] *= 1.3
    base[..., 2] *= 0.7
    img = np.clip(base, 0, 255).astype(np.uint8)
    before = channel_means(img)
    assert before[0] > before[1] > before[2], f"test setup wrong, got {before}"

    out = gray_world(img)
    after = channel_means(out)
    # All three channel means should be within ~1 uint8 step of each other.
    assert max(after) - min(after) <= 1.0, f"means still unbalanced: {after}"
    # Overall luminance should be essentially preserved.
    assert abs(before.mean() - after.mean()) <= 1.5


def test_gray_world_is_idempotent_on_already_neutral_image() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(64, 192, size=(32, 32, 3), dtype=np.uint8)
    once = gray_world(img)
    twice = gray_world(once)
    # Second pass shouldn't shift channel means further.
    m1 = channel_means(once)
    m2 = channel_means(twice)
    assert np.allclose(m1, m2, atol=1.0)


def test_gray_world_handles_pitch_black_image() -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = gray_world(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert np.array_equal(out, img)  # no scaling applied


def test_gray_world_accepts_float_input_and_preserves_range() -> None:
    rng = np.random.default_rng(2)
    img = rng.uniform(0.0, 1.0, size=(16, 16, 3)).astype(np.float32)
    img[..., 0] *= 1.2  # warm cast
    img = np.clip(img, 0.0, 1.0)
    out = gray_world(img)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    m = channel_means(out)
    assert max(m) - min(m) < 0.02


def test_gray_world_rejects_non_rgb() -> None:
    with pytest.raises(ValueError):
        gray_world(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        gray_world(np.zeros((8, 8, 4), dtype=np.uint8))


# ---------- Debug grids ----------------------------------------------------


def _fake_frame(rng: np.random.Generator, cast: tuple[float, float, float]) -> np.ndarray:
    base = rng.integers(64, 192, size=(128, 192, 3), dtype=np.uint8).astype(np.float64)
    for c, k in enumerate(cast):
        base[..., c] *= k
    return np.clip(base, 0, 255).astype(np.uint8)


def test_save_debug_grid_writes_expected_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    # 25 frames with varying casts → should produce 3 grids (10 + 10 + 5).
    frames = []
    for i in range(25):
        cast = (1.0 + 0.01 * i, 1.0, 1.0 - 0.01 * i)
        before = _fake_frame(rng, cast)
        after = gray_world(before)
        frames.append((f"frame_{i:03d}", before, after))

    out_dir = tmp_path / "wb"
    written = save_whitebalance_debug_grid(frames, out_dir, rows_per_grid=10)
    assert [p.name for p in written] == [
        "whitebalance_00.png",
        "whitebalance_01.png",
        "whitebalance_02.png",
    ]
    for p in written:
        assert p.exists() and p.stat().st_size > 0
        # Sanity: each grid opens as a valid PNG.
        with Image.open(p) as im:
            assert im.mode == "RGB"
            assert im.size[0] > 0 and im.size[1] > 0


def test_save_debug_grid_handles_empty_iterable(tmp_path: Path) -> None:
    out_dir = tmp_path / "wb_empty"
    written = save_whitebalance_debug_grid([], out_dir)
    assert written == []
    # Directory is still created (caller may chain writes to it later).
    assert out_dir.exists()
