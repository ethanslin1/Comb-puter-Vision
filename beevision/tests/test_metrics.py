"""Tests for the colony-health metrics module."""
from __future__ import annotations

from dataclasses import asdict

import math

import pytest

from beevision.metrics import (
    BeeInstance,
    BroodMetrics,
    CellInstance,
    CompositionMetrics,
    HealthReport,
    MiteMetrics,
    ScoreWeights,
    brood_health_score,
    build_health_report,
    compute_brood_metrics,
    compute_composition_metrics,
    compute_mite_metrics,
    composite_score,
)


# ---------- Type validation ------------------------------------------------


def test_cell_instance_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="unknown cell class"):
        CellInstance(cls="not_a_class", position=(0.0, 0.0))


def test_cell_instance_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="score must be in"):
        CellInstance(cls="capped_brood", position=(0.0, 0.0), score=1.5)


def test_bee_instance_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="unknown mite class"):
        BeeInstance(cls="bee", position=(0.0, 0.0))


# ---------- Brood metrics --------------------------------------------------


def _grid_brood(n_per_side: int, spacing: float = 1.0, cls: str = "capped_brood") -> list[CellInstance]:
    """Return an n×n grid of brood cells; their NN distances are uniform."""
    return [
        CellInstance(cls=cls, position=(i * spacing, j * spacing))
        for i in range(n_per_side)
        for j in range(n_per_side)
    ]


def test_brood_empty_inputs_returns_zeros_no_nan() -> None:
    m = compute_brood_metrics([])
    assert m.n_brood_cells == 0 and m.n_total_cells == 0
    assert m.brood_fraction == 0.0
    assert m.fill_ratio == 0.0
    assert m.nearest_neighbor_cv == 0.0
    assert m.regularity == 0.0
    assert m.capped_brood_fraction == 0.0


def test_brood_perfect_grid_high_regularity() -> None:
    m = compute_brood_metrics(_grid_brood(5))  # 25 cells in 5x5 grid
    # Uniform NN spacing → CV ≈ 0.
    assert m.nearest_neighbor_cv == pytest.approx(0.0, abs=1e-6)
    assert m.fill_ratio > 0.5     # densely packed within hull
    assert m.regularity > 0.5
    assert m.capped_brood_fraction == pytest.approx(1.0)


def test_brood_scattered_pattern_lower_regularity_than_grid() -> None:
    grid_m = compute_brood_metrics(_grid_brood(5))
    # Scatter: same brood cells but with one isolated outlier.
    scattered = _grid_brood(5)
    scattered.append(CellInstance(cls="capped_brood", position=(100.0, 100.0)))
    scattered_m = compute_brood_metrics(scattered)
    assert scattered_m.regularity < grid_m.regularity
    # The outlier dramatically inflates hull area → fill_ratio drops.
    assert scattered_m.fill_ratio < grid_m.fill_ratio


def test_brood_capped_fraction_correct() -> None:
    cells = [
        CellInstance(cls="capped_brood", position=(0.0, 0.0)),
        CellInstance(cls="capped_brood", position=(1.0, 0.0)),
        CellInstance(cls="larva", position=(0.0, 1.0)),
        CellInstance(cls="egg", position=(1.0, 1.0)),
    ]
    m = compute_brood_metrics(cells)
    assert m.n_brood_cells == 4
    assert m.capped_brood_fraction == pytest.approx(0.5)


def test_brood_fraction_against_total() -> None:
    cells = [
        *_grid_brood(2),  # 4 brood cells
        CellInstance(cls="honey", position=(10.0, 0.0)),
        CellInstance(cls="honey", position=(11.0, 0.0)),
    ]
    m = compute_brood_metrics(cells)
    assert m.n_brood_cells == 4
    assert m.n_total_cells == 6
    assert m.brood_fraction == pytest.approx(4 / 6)


def test_brood_single_cell_no_nan() -> None:
    m = compute_brood_metrics([CellInstance(cls="capped_brood", position=(0.0, 0.0))])
    # Single point: hull area = 0, NN distances empty.
    assert m.fill_ratio == 0.0
    assert m.nearest_neighbor_cv == 0.0
    assert m.regularity == 0.0
    assert m.capped_brood_fraction == 1.0


# ---------- Composition metrics --------------------------------------------


def test_composition_empty_inputs() -> None:
    m = compute_composition_metrics([])
    assert m.n_food_cells == 0
    assert m.honey_pollen_ratio is None
    assert m.food_balance == 0.0


def test_composition_counts_each_food_class() -> None:
    cells = [
        CellInstance(cls="honey", position=(0.0, 0.0)),
        CellInstance(cls="honey", position=(1.0, 0.0)),
        CellInstance(cls="honey", position=(2.0, 0.0)),
        CellInstance(cls="nectar", position=(3.0, 0.0)),
        CellInstance(cls="pollen", position=(4.0, 0.0)),
        CellInstance(cls="capped_brood", position=(5.0, 0.0)),
    ]
    m = compute_composition_metrics(cells)
    assert m.n_honey == 3 and m.n_nectar == 1 and m.n_pollen == 1
    assert m.n_food_cells == 5
    assert m.food_fraction == pytest.approx(5 / 6)
    assert m.honey_pollen_ratio == pytest.approx(3.0)


def test_composition_balance_peaks_at_70_percent_honey() -> None:
    """7 honey + 3 pollen = ratio 7/3, p=0.7 → balance ≈ 1."""
    cells = [CellInstance(cls="honey", position=(i, 0)) for i in range(7)]
    cells += [CellInstance(cls="pollen", position=(i, 1)) for i in range(3)]
    m = compute_composition_metrics(cells)
    assert m.food_balance == pytest.approx(1.0, abs=1e-6)


def test_composition_balance_drops_at_extremes() -> None:
    # All honey, no pollen → far from 0.7, balance much less than 1.
    cells = [CellInstance(cls="honey", position=(i, 0)) for i in range(10)]
    m = compute_composition_metrics(cells)
    assert m.honey_pollen_ratio is None
    assert m.food_balance < 0.5


def test_composition_no_pollen_returns_none_ratio() -> None:
    cells = [CellInstance(cls="honey", position=(0.0, 0.0))]
    m = compute_composition_metrics(cells)
    assert m.honey_pollen_ratio is None


# ---------- Mite metrics --------------------------------------------------


def test_mites_empty_inputs() -> None:
    m = compute_mite_metrics([])
    assert m.n_bees == 0 and m.n_mite == 0
    assert m.mite_load == 0.0
    assert m.mite_load_weighted == 0.0


def test_mites_hard_load_matches_count() -> None:
    bees = [
        BeeInstance(cls="no_mite", position=(0.0, 0.0)),
        BeeInstance(cls="no_mite", position=(1.0, 0.0)),
        BeeInstance(cls="mite", position=(2.0, 0.0)),
    ]
    m = compute_mite_metrics(bees)
    assert m.n_mite == 1
    assert m.mite_load == pytest.approx(1 / 3)


def test_mites_soft_load_uses_probabilities_when_present() -> None:
    bees = [
        BeeInstance(cls="no_mite", position=(0.0, 0.0), score=0.1),
        BeeInstance(cls="no_mite", position=(1.0, 0.0), score=0.2),
        BeeInstance(cls="mite", position=(2.0, 0.0), score=0.9),
    ]
    m = compute_mite_metrics(bees)
    # Soft load = mean of [0.1, 0.2, 0.9] = 0.4.
    assert m.mite_load_weighted == pytest.approx(0.4)
    # Hard load is unaffected.
    assert m.mite_load == pytest.approx(1 / 3)


def test_mites_soft_load_falls_back_to_hard_label_without_score() -> None:
    bees = [
        BeeInstance(cls="no_mite", position=(0.0, 0.0)),  # no score → 0
        BeeInstance(cls="mite", position=(1.0, 0.0)),     # no score → 1
    ]
    m = compute_mite_metrics(bees)
    assert m.mite_load_weighted == pytest.approx(0.5)


# ---------- Composite score ----------------------------------------------


def _ideal_health_inputs() -> tuple[list[CellInstance], list[BeeInstance]]:
    """Build inputs that should score near 10."""
    cells: list[CellInstance] = []
    # Tight 5x5 capped-brood grid.
    cells += _grid_brood(5)
    # Healthy honey:pollen ratio ≈ 7:3.
    cells += [CellInstance(cls="honey", position=(20 + i, 0.0)) for i in range(7)]
    cells += [CellInstance(cls="pollen", position=(30 + i, 0.0)) for i in range(3)]
    # 10 bees, 0 mites.
    bees = [BeeInstance(cls="no_mite", position=(0.0, i), score=0.05) for i in range(10)]
    return cells, bees


def _failing_health_inputs() -> tuple[list[CellInstance], list[BeeInstance]]:
    """Build inputs that should score near 1."""
    # No brood, no food.
    cells: list[CellInstance] = [
        CellInstance(cls="other", position=(0.0, 0.0)),
        CellInstance(cls="other", position=(1.0, 0.0)),
    ]
    # Heavy mite load.
    bees = [BeeInstance(cls="mite", position=(0.0, i), score=0.95) for i in range(10)]
    return cells, bees


def test_composite_score_in_range() -> None:
    brood = compute_brood_metrics([])
    comp = compute_composition_metrics([])
    mite = compute_mite_metrics([])
    score, _, _ = composite_score(brood, comp, mite)
    assert 1.0 <= score <= 10.0


def test_composite_score_ideal_is_high() -> None:
    cells, bees = _ideal_health_inputs()
    report = build_health_report(cells, bees)
    assert report.composite_score >= 8.5
    # Components should also be strong.
    assert report.score_components["brood_health"] > 0.5
    assert report.score_components["food_balance"] > 0.95
    assert report.score_components["mite_safety"] > 0.9


def test_composite_score_failing_is_low() -> None:
    cells, bees = _failing_health_inputs()
    report = build_health_report(cells, bees)
    assert report.composite_score <= 2.0
    assert report.score_components["brood_health"] == pytest.approx(0.0)
    assert report.score_components["food_balance"] == pytest.approx(0.0)
    assert report.score_components["mite_safety"] < 0.1


def test_composite_score_weights_normalize() -> None:
    weights = ScoreWeights(brood=2.0, food=1.0, mites=1.0)  # not summing to 1
    n = weights.normalized()
    assert n.brood + n.food + n.mites == pytest.approx(1.0)
    assert n.brood == pytest.approx(0.5)


def test_composite_score_weights_invalid() -> None:
    with pytest.raises(ValueError):
        ScoreWeights(brood=0.0, food=0.0, mites=0.0).normalized()


def test_composite_score_brood_weight_dominates_when_set() -> None:
    """Sanity: if I crank brood weight to 1.0, score follows brood signal."""
    cells, bees = _ideal_health_inputs()
    # Replace bees with all-mite to tank the mite axis.
    bees = [BeeInstance(cls="mite", position=(0.0, i), score=0.95) for i in range(10)]
    # Default weights → score should drop hard.
    report_default = build_health_report(cells, bees)
    # Brood-only weights → mite axis ignored.
    report_brood = build_health_report(
        cells, bees, weights=ScoreWeights(brood=1.0, food=0.0, mites=0.0)
    )
    assert report_brood.composite_score > report_default.composite_score


# ---------- Health report assembly ----------------------------------------


def test_health_report_serializes_to_dict() -> None:
    cells, bees = _ideal_health_inputs()
    report = build_health_report(cells, bees)
    d = asdict(report)
    assert "composite_score" in d
    assert "brood" in d and "regularity" in d["brood"]
    assert "composition" in d
    assert "mites" in d
    # Components dict round-trips.
    assert "raw" in d["score_components"]


def test_health_report_notes_flag_missing_signals() -> None:
    # No brood, no food, no bees → all three notes.
    report = build_health_report([], [])
    notes = " | ".join(report.notes)
    assert "brood" in notes and "bees" in notes and "food" in notes


def test_brood_health_score_is_geometric_mean() -> None:
    brood = BroodMetrics(
        n_brood_cells=10, n_total_cells=10, brood_fraction=1.0,
        fill_ratio=0.5, nearest_neighbor_cv=0.5, regularity=0.4,
        capped_brood_fraction=0.9,
    )
    # sqrt(0.4 * 0.9) ≈ 0.6
    assert brood_health_score(brood) == pytest.approx(math.sqrt(0.4 * 0.9))


def test_brood_health_score_zero_capped_brood_makes_score_zero() -> None:
    brood = BroodMetrics(
        n_brood_cells=10, n_total_cells=10, brood_fraction=1.0,
        fill_ratio=1.0, nearest_neighbor_cv=0.0, regularity=1.0,
        capped_brood_fraction=0.0,  # nothing has reached capped stage yet
    )
    assert brood_health_score(brood) == pytest.approx(0.0)
