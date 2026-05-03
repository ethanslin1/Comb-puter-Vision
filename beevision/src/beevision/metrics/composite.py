"""Composite 1-10 colony health score.

Combines the three sub-metric blocks (brood, composition, mites) into a
single interpretable score. The scoring formula is intentionally simple and
its weights are exposed as a dataclass so they can be tuned without code
changes:

    raw = w_brood   * brood_health_score(brood)
        + w_food    * composition.food_balance
        + w_mites   * (1 - mites.mite_load_weighted)

    composite = clamp(round(1 + 9 * raw, 1), 1, 10)

where ``brood_health_score`` is the geometric mean of ``brood.regularity``
and ``brood.capped_brood_fraction`` — a frame with zero capped brood gets
zero brood credit no matter how regular the spacing of the eggs/larvae,
because the queen's productivity hasn't been confirmed yet.

Default weights are equal across the three axes (1/3 each). The README's
clinical interpretation:
- 9–10: thriving colony, no concerns
- 6–8 : healthy with minor issues (e.g. mild mite load, slight brood gaps)
- 4–5 : at-risk; intervene
- 1–3 : failing; queen problem, severe disease, or heavy infestation

The function records every component value and weight in the returned
``HealthReport.score_components``/``weights`` dicts so the score is
reproducible and debuggable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beevision.metrics.brood import compute_brood_metrics
from beevision.metrics.composition import compute_composition_metrics
from beevision.metrics.mites import compute_mite_metrics
from beevision.metrics.types import (
    BeeInstance,
    BroodMetrics,
    CellInstance,
    CompositionMetrics,
    HealthReport,
    MiteMetrics,
)


@dataclass
class ScoreWeights:
    """Weights for the three score axes. Re-normalized to sum to 1 if not."""

    brood: float = 1.0 / 3.0
    food: float = 1.0 / 3.0
    mites: float = 1.0 / 3.0

    def normalized(self) -> "ScoreWeights":
        total = self.brood + self.food + self.mites
        if total <= 0:
            raise ValueError("score weights must sum to a positive number")
        return ScoreWeights(
            brood=self.brood / total,
            food=self.food / total,
            mites=self.mites / total,
        )


def brood_health_score(brood: BroodMetrics) -> float:
    """Geometric mean of regularity and capped-fraction, in [0, 1].

    Returns 0 if either is 0 — a frame with great pattern but no capped
    brood is *not* yet a confirmed productive frame.
    """
    return float(np.sqrt(max(brood.regularity, 0.0) * max(brood.capped_brood_fraction, 0.0)))


def composite_score(
    brood: BroodMetrics,
    composition: CompositionMetrics,
    mites: MiteMetrics,
    weights: ScoreWeights | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Map the three sub-metric blocks to a 1-10 score plus its components.

    Returns ``(score, components, weights_dict)``. The score is rounded to
    one decimal place — finer precision is misleading given how noisy the
    upstream classifiers are.
    """
    w = (weights or ScoreWeights()).normalized()

    brood_s = brood_health_score(brood)
    food_s = max(0.0, min(composition.food_balance, 1.0))
    # Mite penalty: 1 - (mite_load_weighted), clamped.
    mite_s = max(0.0, min(1.0 - mites.mite_load_weighted, 1.0))

    raw = w.brood * brood_s + w.food * food_s + w.mites * mite_s
    raw = max(0.0, min(raw, 1.0))
    score = round(1.0 + 9.0 * raw, 1)
    score = max(1.0, min(score, 10.0))

    components = {
        "brood_health": brood_s,
        "food_balance": food_s,
        "mite_safety": mite_s,
        "raw": raw,
    }
    weights_dict = {"brood": w.brood, "food": w.food, "mites": w.mites}
    return float(score), components, weights_dict


def build_health_report(
    cells: list[CellInstance],
    bees: list[BeeInstance],
    weights: ScoreWeights | None = None,
) -> HealthReport:
    """Run all three sub-metric blocks and assemble a ``HealthReport``."""
    brood = compute_brood_metrics(cells)
    composition = compute_composition_metrics(cells)
    mite = compute_mite_metrics(bees)

    score, components, weights_dict = composite_score(brood, composition, mite, weights)

    notes: list[str] = []
    if brood.n_brood_cells == 0:
        notes.append("no brood detected on this frame")
    if mite.n_bees == 0:
        notes.append("no bees detected; mite load uninformative")
    if composition.n_food_cells == 0:
        notes.append("no food cells detected")

    return HealthReport(
        brood=brood,
        composition=composition,
        mites=mite,
        composite_score=score,
        score_components=components,
        weights=weights_dict,
        notes=notes,
    )
