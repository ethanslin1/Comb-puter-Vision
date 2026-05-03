"""Hive-composition metrics: honey/pollen/nectar storage breakdown.

Cells are classified into food (``honey``, ``nectar``, ``pollen``), brood
(handled by ``brood.py``), and ``other``. This module summarizes the food
side: how many cells of each type, the food fraction of the frame, and a
balance score peaked around the natural healthy ratio of stored honey to
stored pollen.

A healthy hive going into winter typically targets roughly 70% of its non-
brood storage area as honey and 30% as pollen — pollen-heavy frames mean
the colony is short on overwintering carbohydrates; honey-heavy frames
with no pollen mean the colony will struggle to feed brood in spring.
The ``food_balance`` metric is a Gaussian-bell around 0.7 honey-fraction
(of honey + pollen, ignoring nectar which is just immature honey):

    balance(p) = exp(-((p - 0.7) / 0.25)²),   p = honey / (honey + pollen)

This peaks at p=0.7, has half-max around p∈[0.5, 0.9], and tails toward 0
at the extremes. ``None`` is returned for ``honey_pollen_ratio`` when
``pollen == 0`` (rather than emit ``inf`` and surprise downstream code).
"""
from __future__ import annotations

import math

from beevision.metrics.types import (
    FOOD_CLASSES,
    CellInstance,
    CompositionMetrics,
)


def compute_composition_metrics(cells: list[CellInstance]) -> CompositionMetrics:
    """Compute food-storage breakdown metrics from per-cell predictions."""
    n_total = len(cells)
    if n_total == 0:
        return CompositionMetrics(
            n_food_cells=0,
            n_honey=0,
            n_nectar=0,
            n_pollen=0,
            food_fraction=0.0,
            honey_pollen_ratio=None,
            food_balance=0.0,
        )

    counts = {cls: 0 for cls in FOOD_CLASSES}
    for c in cells:
        if c.cls in counts:
            counts[c.cls] += 1

    n_honey = counts["honey"]
    n_nectar = counts["nectar"]
    n_pollen = counts["pollen"]
    n_food = n_honey + n_nectar + n_pollen

    food_fraction = float(n_food / n_total) if n_total else 0.0

    if n_pollen == 0:
        ratio: float | None = None
    else:
        ratio = float(n_honey / n_pollen)

    if (n_honey + n_pollen) == 0:
        balance = 0.0
    else:
        p = n_honey / (n_honey + n_pollen)
        balance = float(math.exp(-((p - 0.7) / 0.25) ** 2))

    return CompositionMetrics(
        n_food_cells=n_food,
        n_honey=n_honey,
        n_nectar=n_nectar,
        n_pollen=n_pollen,
        food_fraction=food_fraction,
        honey_pollen_ratio=ratio,
        food_balance=balance,
    )
