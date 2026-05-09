"""Brood-pattern regularity metrics.

A healthy queen lays eggs in a tight, contiguous, regular pattern across the
brood-nest area; an irregular or "shotgun" brood pattern signals a failing
queen, brood disease, or chilled brood. This module quantifies that pattern
from per-cell predictions on a single frame.

Three sub-metrics, combined into a single ``regularity`` score in [0, 1]:

1. **Fill ratio** — brood-cell count divided by the convex-hull area
   (cells per unit area). High = brood cells densely fill their footprint;
   low = sparse, scattered. Computed in *cell-spacing units* by dividing
   convex-hull area by the median nearest-neighbor distance squared, so
   the metric is image-resolution-independent.

2. **Nearest-neighbor CV** — coefficient of variation (std/mean) of each
   brood cell's distance to its nearest brood neighbor. Low CV = uniform
   spacing (regular pattern). High CV = clustered + isolated cells.

3. **Capped fraction** — capped_brood / total brood. Brood that's reached
   the capped stage is the strongest signal of a productive queen; eggs
   alone could be misleading (they hatch in 3 days).

The composite ``regularity`` is the geometric mean of fill and (1-CV)
clamped to [0, 1], so any single bad signal pulls the score down (a frame
with great fill but irregular spacing is *not* a healthy brood pattern).

Edge cases: 0 brood cells → all zeros, no NaN. 1 brood cell → fill=0
(no convex hull area), CV undefined → set to 0; the resulting regularity
is 0 which correctly reflects "essentially no brood." 2 brood cells → CV
trivially 0 since there's only one NN distance per side; we set
``fill_ratio`` to 0 (line has no area) and ``regularity`` to 0.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from beevision.metrics.types import (
    BROOD_CLASSES,
    BroodMetrics,
    CellInstance,
)


def _convex_hull_area(points: np.ndarray) -> float:
    """Area of the 2D convex hull, monotone-chain style.

    Returns 0 for fewer than 3 points or co-linear points.
    """
    if len(points) < 3:
        return 0.0
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    # Build lower hull.
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """Per-point distance to its nearest neighbor in ``points``.

    O(n²); fine for the cell counts we expect (~hundreds per frame).
    """
    n = len(points)
    if n < 2:
        return np.empty(0, dtype=np.float64)
    diffs = points[:, None, :] - points[None, :, :]  
    dists = np.linalg.norm(diffs, axis=-1)          
    np.fill_diagonal(dists, np.inf)
    return dists.min(axis=1)


def _filter_brood(cells: Iterable[CellInstance]) -> list[CellInstance]:
    return [c for c in cells if c.cls in BROOD_CLASSES]


def compute_brood_metrics(cells: list[CellInstance]) -> BroodMetrics:
    """Compute brood-pattern metrics from per-cell predictions on a frame."""
    n_total = len(cells)
    brood = _filter_brood(cells)
    n_brood = len(brood)

    if n_total == 0 or n_brood == 0:
        return BroodMetrics(
            n_brood_cells=n_brood,
            n_total_cells=n_total,
            brood_fraction=0.0,
            fill_ratio=0.0,
            nearest_neighbor_cv=0.0,
            regularity=0.0,
            capped_brood_fraction=0.0,
        )

    pts = np.array([c.position for c in brood], dtype=np.float64)
    nn = _nearest_neighbor_distances(pts)

    if len(nn) == 0 or nn.mean() == 0:
        nn_cv = 0.0
    else:
        nn_cv = float(nn.std() / nn.mean())

    hull_area = _convex_hull_area(pts)
    if hull_area > 0 and len(nn) > 0:
        median_nn = float(np.median(nn))
        if median_nn > 0:
            cells_per_unit = n_brood * (median_nn ** 2) / hull_area
            fill_ratio = float(min(cells_per_unit / 1.155, 1.0))
        else:
            fill_ratio = 0.0
    else:
        fill_ratio = 0.0

    spacing_score = max(0.0, 1.0 - min(nn_cv, 1.0))
    regularity = float(np.sqrt(fill_ratio * spacing_score))

    capped = sum(1 for c in brood if c.cls == "capped_brood")
    capped_fraction = float(capped / n_brood)

    return BroodMetrics(
        n_brood_cells=n_brood,
        n_total_cells=n_total,
        brood_fraction=float(n_brood / n_total),
        fill_ratio=fill_ratio,
        nearest_neighbor_cv=nn_cv,
        regularity=regularity,
        capped_brood_fraction=capped_fraction,
    )
