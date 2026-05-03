"""Colony-health metrics module.

Pure-NumPy aggregation over per-cell + per-bee predictions on a single
frame. Independent of torch, models, geometry — operates only on the
``CellInstance``/``BeeInstance`` records the pipeline assembles.

Typical use::

    from beevision.metrics import build_health_report
    report = build_health_report(cells, bees)
    print(report.composite_score)        # 1-10
    print(report.brood.regularity)       # 0-1
    json.dump(asdict(report), fh)        # serialize to disk
"""
from beevision.metrics.brood import compute_brood_metrics
from beevision.metrics.composite import (
    ScoreWeights,
    brood_health_score,
    build_health_report,
    composite_score,
)
from beevision.metrics.composition import compute_composition_metrics
from beevision.metrics.mites import compute_mite_metrics
from beevision.metrics.types import (
    BROOD_CLASSES,
    FOOD_CLASSES,
    OTHER_CLASSES,
    BeeInstance,
    BroodMetrics,
    CellInstance,
    CompositionMetrics,
    HealthReport,
    MiteMetrics,
    bees_from_records,
    cells_from_records,
)

__all__ = [
    # types
    "CellInstance",
    "BeeInstance",
    "BroodMetrics",
    "CompositionMetrics",
    "MiteMetrics",
    "HealthReport",
    "BROOD_CLASSES",
    "FOOD_CLASSES",
    "OTHER_CLASSES",
    "cells_from_records",
    "bees_from_records",
    # computations
    "compute_brood_metrics",
    "compute_composition_metrics",
    "compute_mite_metrics",
    "composite_score",
    "brood_health_score",
    "build_health_report",
    "ScoreWeights",
]
