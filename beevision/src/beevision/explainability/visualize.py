"""Colormap + overlay helpers for explainability heatmaps.

We avoid a hard matplotlib runtime dependency for inference: implement the
``jet`` and ``inferno`` colormaps directly. These are sufficient for
saving overlay images to disk; a Streamlit app can use matplotlib directly
if richer colormaps are desired.

Inputs throughout are numpy arrays — heatmaps are ``(H, W)`` float in
[0, 1] (or signed; ``apply_signed_colormap`` handles that). Image inputs
are either ``(H, W, 3)`` uint8 or torch tensors that the caller has
already converted.
"""
from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger("explainability.visualize")


# ---------- Colormaps ------------------------------------------------------


def _jet_lut() -> np.ndarray:
    """256-entry jet RGB LUT, uint8."""
    # Standard piecewise-linear jet definition.
    n = 256
    t = np.linspace(0.0, 1.0, n)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def _inferno_lut() -> np.ndarray:
    """256-entry inferno RGB LUT, uint8 (matplotlib-style approximation)."""
    # Anchor stops sampled from the matplotlib inferno colormap.
    stops = np.array([
        [0, 0, 4],
        [40, 11, 84],
        [101, 21, 110],
        [159, 42, 99],
        [212, 72, 66],
        [245, 125, 21],
        [250, 193, 39],
        [252, 255, 164],
    ], dtype=np.float32)
    xs = np.linspace(0.0, 1.0, len(stops))
    ts = np.linspace(0.0, 1.0, 256)
    out = np.empty((256, 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = np.interp(ts, xs, stops[:, c])
    return np.clip(out, 0, 255).astype(np.uint8)


_LUTS = {
    "jet": _jet_lut(),
    "inferno": _inferno_lut(),
}


def apply_colormap(heatmap: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """Map a [0, 1] grayscale heatmap to an (H, W, 3) uint8 RGB image."""
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D; got shape {heatmap.shape}")
    if cmap not in _LUTS:
        raise ValueError(f"unknown colormap {cmap!r}; choose from {sorted(_LUTS)}")
    lut = _LUTS[cmap]
    h = np.clip(heatmap, 0.0, 1.0)
    idx = (h * 255).astype(np.int64)
    return lut[idx]  # (H, W, 3) uint8


def apply_signed_colormap(heatmap: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """Map a signed heatmap to RGB by normalizing |x| / max|x|.

    LIME-style coefficients can be negative; this rescales to [0, 1] before
    coloring. The sign information is lost — for two-tone (red/blue)
    visualization, threshold and color the halves separately upstream.
    """
    if heatmap.size == 0:
        return apply_colormap(np.zeros_like(heatmap, dtype=np.float32), cmap)
    m = float(np.max(np.abs(heatmap)))
    if m <= 0:
        return apply_colormap(np.zeros_like(heatmap, dtype=np.float32), cmap)
    norm = np.abs(heatmap) / m
    return apply_colormap(norm.astype(np.float32), cmap)


# ---------- Overlay --------------------------------------------------------


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    cmap: str = "jet",
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a [0, 1] heatmap on top of an RGB image.

    ``image_rgb`` is ``(H, W, 3)`` uint8 (e.g. from PIL → np.array).
    Heatmap is bilinearly resized to match if shapes differ. Alpha is the
    heatmap weight in [0, 1].
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be (H, W, 3); got {image_rgb.shape}")
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D; got {heatmap.shape}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")

    H, W, _ = image_rgb.shape
    if heatmap.shape != (H, W):
        heatmap = _bilinear_resize(heatmap, H, W)
    color = apply_colormap(heatmap, cmap=cmap).astype(np.float32)
    base = image_rgb.astype(np.float32)
    blended = base * (1.0 - alpha) + color * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _bilinear_resize(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Tiny bilinear resize of a 2D array using numpy only."""
    h, w = arr.shape
    if (h, w) == (H, W):
        return arr
    ys = np.linspace(0, h - 1, H, dtype=np.float64)
    xs = np.linspace(0, w - 1, W, dtype=np.float64)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (ys - y0).reshape(-1, 1)
    fx = (xs - x0).reshape(1, -1)

    a = arr[y0[:, None], x0[None, :]]
    b = arr[y0[:, None], x1[None, :]]
    c = arr[y1[:, None], x0[None, :]]
    d = arr[y1[:, None], x1[None, :]]
    top = a * (1 - fx) + b * fx
    bot = c * (1 - fx) + d * fx
    return (top * (1 - fy) + bot * fy).astype(arr.dtype)
