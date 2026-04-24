"""Gray-world white balance for comb frames.

Spec rule #2: "sRGB + gray-world white balance on frames (not crops). Save
debug grids for first 50 frames in interim/_debug/whitebalance/."

The raw DS-COMB frames were shot inside hives under mixed daylight + flash
and vary significantly in color cast. Before we save the normalized PNG,
we scale each channel so its mean equals the overall frame mean — the
classic gray-world assumption that "the average of a natural scene is
achromatic."

Caveat: we apply gray-world directly in sRGB space rather than converting
to linear light first. For 8-bit photographic input the difference is
within a few LSBs, and keeping it in sRGB means the decoder + encoder do
not need a round-trip gamma op per frame. The docstring records this
choice so that if a downstream color-critical task ever needs it, we can
flip on a ``linearize=True`` variant without changing callers.

Only frames get white-balanced (spec: "not crops") — the cell-crop
ingests (deepbee_cls, varroa, beeimage) must not call these helpers.

Debug artifacts
---------------
``save_whitebalance_debug_grid(frames, out_dir)`` writes one or more
PNG grids under ``interim/_debug/whitebalance/`` with side-by-side
before / after thumbnails for auditing. 50 frames at 10 per grid
yields ``whitebalance_00.png`` … ``whitebalance_04.png``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "gray_world",
    "channel_means",
    "save_whitebalance_debug_grid",
]

LOG = logging.getLogger("whitebalance")

# Debug grid geometry.
_GRID_ROWS = 10                 # frames per grid image
_GRID_THUMB = 256               # per-thumbnail side in px (short edge)
_GRID_GAP = 6                   # pixels between before/after
_GRID_MARGIN = 6                # outer margin
_GRID_LABEL_H = 18              # room for caption under each row


# ---------- Core algorithm --------------------------------------------------


def channel_means(img: np.ndarray) -> np.ndarray:
    """Return the per-channel mean of a (H, W, 3) uint8/float image as float64."""
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {img.shape}")
    return img.reshape(-1, 3).mean(axis=0).astype(np.float64)


def gray_world(img: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Gray-world white balance, per-channel rescale to the overall mean.

    Accepts uint8 HWC RGB (the common case) and returns uint8 HWC RGB.
    Also accepts float inputs in [0, 1] and returns float in the same range.

    A flat/grayscale frame with zero-variance channels is returned unchanged
    (no scaling makes sense, division-by-zero protected by ``eps``).
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {img.shape}")
    float_input = np.issubdtype(img.dtype, np.floating)
    arr = img.astype(np.float64, copy=False)
    means = arr.reshape(-1, 3).mean(axis=0)
    overall = float(means.mean())
    if overall < eps:
        # Image is ~pitch black; scaling is meaningless, return a copy.
        return img.copy()
    scale = overall / np.maximum(means, eps)
    out = arr * scale
    if float_input:
        return np.clip(out, 0.0, 1.0).astype(img.dtype, copy=False)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ---------- Debug grids -----------------------------------------------------


def _fit_thumb(img: Image.Image, target: int) -> Image.Image:
    """Resize so the short edge == target, preserving aspect ratio."""
    w, h = img.size
    s = min(w, h)
    if s == target:
        out = img
    else:
        scale = target / float(s)
        out = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
    # Center-crop to a square for a tidy grid; these are debug thumbs.
    nw, nh = out.size
    left = (nw - target) // 2
    top = (nh - target) // 2
    return out.crop((left, top, left + target, top + target))


def _try_font(size: int) -> ImageFont.ImageFont:
    # Matplotlib ships a DejaVuSans; we look for it. Fall back to default bitmap.
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001
                pass
    return ImageFont.load_default()


def _compose_grid(
    items: Sequence[tuple[str, Image.Image, Image.Image, np.ndarray, np.ndarray]],
    title: str,
) -> Image.Image:
    """Layout: N rows × 2 cols (before | after), caption under each row."""
    rows = len(items)
    cell_w = _GRID_THUMB
    cell_h = _GRID_THUMB + _GRID_LABEL_H
    w = _GRID_MARGIN * 2 + cell_w * 2 + _GRID_GAP
    h = _GRID_MARGIN * 2 + cell_h * rows + 26  # +title strip
    canvas = Image.new("RGB", (w, h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = _try_font(12)
    title_font = _try_font(14)

    draw.text((_GRID_MARGIN, 4), title, fill=(230, 230, 230), font=title_font)

    y0 = 26 + _GRID_MARGIN
    for i, (name, before_img, after_img, before_mean, after_mean) in enumerate(items):
        row_y = y0 + i * cell_h
        canvas.paste(before_img, (_GRID_MARGIN, row_y))
        canvas.paste(after_img, (_GRID_MARGIN + cell_w + _GRID_GAP, row_y))
        label = (
            f"{name}    before RGB={before_mean[0]:.0f},{before_mean[1]:.0f},"
            f"{before_mean[2]:.0f}    after RGB={after_mean[0]:.0f},"
            f"{after_mean[1]:.0f},{after_mean[2]:.0f}"
        )
        draw.text(
            (_GRID_MARGIN, row_y + _GRID_THUMB + 2),
            label,
            fill=(210, 210, 210),
            font=font,
        )
    return canvas


def save_whitebalance_debug_grid(
    frames: Iterable[tuple[str, np.ndarray, np.ndarray]],
    out_dir: Path,
    *,
    rows_per_grid: int = _GRID_ROWS,
    prefix: str = "whitebalance",
) -> list[Path]:
    """Write before/after grids under ``out_dir``.

    ``frames`` yields ``(name, before_uint8_hwc, after_uint8_hwc)`` triples.
    Emits ``<prefix>_00.png``, ``<prefix>_01.png``, … with ``rows_per_grid``
    frames each. Returns the list of written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch: list[tuple[str, Image.Image, Image.Image, np.ndarray, np.ndarray]] = []
    written: list[Path] = []

    def _flush() -> None:
        if not batch:
            return
        grid_idx = len(written)
        title = f"gray-world white balance — grid {grid_idx:02d} ({len(batch)} frames)"
        img = _compose_grid(batch, title)
        out_path = out_dir / f"{prefix}_{grid_idx:02d}.png"
        img.save(out_path)
        written.append(out_path)
        batch.clear()

    for name, before, after in frames:
        b_thumb = _fit_thumb(Image.fromarray(before, mode="RGB"), _GRID_THUMB)
        a_thumb = _fit_thumb(Image.fromarray(after, mode="RGB"), _GRID_THUMB)
        b_mean = channel_means(before)
        a_mean = channel_means(after)
        batch.append((name, b_thumb, a_thumb, b_mean, a_mean))
        if len(batch) >= rows_per_grid:
            _flush()
    _flush()
    LOG.info("wrote %d white-balance debug grid(s) → %s", len(written), out_dir)
    return written
