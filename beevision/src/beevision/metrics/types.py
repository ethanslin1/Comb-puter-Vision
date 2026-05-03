"""Inference-time record types for the metrics + pipeline modules.

These are distinct from ``data/schema.py``'s Pydantic training-time records
(``CellRecord``, ``MiteRecord``, ``FrameRecord``) — those describe ground-
truth dataset rows. The types here describe *predictions* over a single
unseen frame: cells the cell classifier emitted, bees the mite classifier
scored, and the aggregated ``HealthReport`` that gets serialized to JSON.

Plain dataclasses, not Pydantic — no I/O validation needed (pipeline code
constructs these directly), and ``dataclasses.asdict()`` serializes cleanly
for the JSON report.

Class-name groupings used by the metrics:

- ``BROOD_CLASSES``  = (egg, larva, capped_brood)
- ``FOOD_CLASSES``   = (honey, nectar, pollen)
- ``OTHER_CLASSES``  = (other,)

These align with ``CELL_CLASSES`` in ``data/datasets.py``. If that tuple
ever changes, update ``_validate_cell_class_groupings`` below — it asserts
the unions partition ``CELL_CLASSES`` exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from beevision.data.datasets import CELL_CLASSES, MITE_CLASSES

# ---------- Class groupings ------------------------------------------------

BROOD_CLASSES: tuple[str, ...] = ("egg", "larva", "capped_brood")
FOOD_CLASSES: tuple[str, ...] = ("honey", "nectar", "pollen")
OTHER_CLASSES: tuple[str, ...] = ("other",)


def _validate_cell_class_groupings() -> None:
    """Assert the brood/food/other groupings partition ``CELL_CLASSES`` exactly."""
    union = set(BROOD_CLASSES) | set(FOOD_CLASSES) | set(OTHER_CLASSES)
    cell = set(CELL_CLASSES)
    if union != cell:
        missing = cell - union
        extra = union - cell
        raise RuntimeError(
            f"BROOD/FOOD/OTHER groupings out of sync with CELL_CLASSES; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


# Validate at import time so any future schema drift fails loudly.
_validate_cell_class_groupings()


# ---------- Per-instance prediction records --------------------------------


@dataclass(frozen=True)
class CellInstance:
    """One cell on a frame: predicted class + position + (optional) confidence.

    ``cls`` must be one of ``CELL_CLASSES``. ``position`` is the cell's
    centroid in image coordinates, ``(x, y)`` in pixels. ``score`` is the
    softmax probability the classifier assigned to ``cls`` (in [0, 1]); it
    is ``None`` if the upstream code dropped the per-instance probability.
    """

    cls: str
    position: tuple[float, float]
    score: float | None = None

    def __post_init__(self) -> None:
        if self.cls not in CELL_CLASSES:
            raise ValueError(
                f"unknown cell class {self.cls!r}; expected one of {CELL_CLASSES}"
            )
        if self.score is not None and not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1] or None; got {self.score}")


@dataclass(frozen=True)
class BeeInstance:
    """One bee on a frame: mite/no_mite prediction + (optional) confidence.

    ``cls`` must be one of ``MITE_CLASSES`` (``no_mite`` or ``mite``).
    ``position`` is the bee's centroid in image coordinates. ``score`` is
    the softmax probability for the *positive* (``mite``) class; for a
    ``no_mite`` prediction the score will typically be < 0.5.
    """

    cls: str
    position: tuple[float, float]
    score: float | None = None

    def __post_init__(self) -> None:
        if self.cls not in MITE_CLASSES:
            raise ValueError(
                f"unknown mite class {self.cls!r}; expected one of {MITE_CLASSES}"
            )
        if self.score is not None and not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1] or None; got {self.score}")


# ---------- Aggregated outputs --------------------------------------------


@dataclass
class BroodMetrics:
    """Brood-pattern health summary."""

    n_brood_cells: int
    n_total_cells: int
    brood_fraction: float                  # n_brood / n_total
    fill_ratio: float                      # brood / convex_hull(brood) area, in [0, 1]
    nearest_neighbor_cv: float             # coefficient of variation of NN distances
    regularity: float                      # composite [0, 1]: 1 = tight, regular pattern
    capped_brood_fraction: float           # capped_brood / brood (queen productivity)


@dataclass
class CompositionMetrics:
    """Honey/pollen/food storage breakdown."""

    n_food_cells: int
    n_honey: int
    n_nectar: int
    n_pollen: int
    food_fraction: float                   # food / total_cells
    honey_pollen_ratio: float | None       # honey / pollen, ``None`` if pollen=0
    food_balance: float                    # [0, 1]: peaked when honey:pollen near 70:30


@dataclass
class MiteMetrics:
    """Mite infestation summary."""

    n_bees: int
    n_mite: int
    mite_load: float                       # n_mite / n_bees, in [0, 1]
    mite_load_weighted: float              # mean of positive-class probabilities, in [0, 1]


@dataclass
class HealthReport:
    """End-to-end health summary produced by the pipeline.

    ``composite_score`` is the headline 1–10 colony health score. The other
    fields are the per-axis breakdowns that fed it; reporting them keeps the
    score interpretable rather than a black-box number.
    """

    brood: BroodMetrics
    composition: CompositionMetrics
    mites: MiteMetrics
    composite_score: float                 # [1, 10]
    score_components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------- Convenience constructors --------------------------------------


def cells_from_records(records: list[dict[str, Any]]) -> list[CellInstance]:
    """Build a ``CellInstance`` list from a list of dicts (e.g. JSON rows).

    Each dict needs ``cls`` and ``position``; ``score`` is optional.
    """
    return [
        CellInstance(
            cls=str(r["cls"]),
            position=(float(r["position"][0]), float(r["position"][1])),
            score=float(r["score"]) if r.get("score") is not None else None,
        )
        for r in records
    ]


def bees_from_records(records: list[dict[str, Any]]) -> list[BeeInstance]:
    """Build a ``BeeInstance`` list from a list of dicts."""
    return [
        BeeInstance(
            cls=str(r["cls"]),
            position=(float(r["position"][0]), float(r["position"][1])),
            score=float(r["score"]) if r.get("score") is not None else None,
        )
        for r in records
    ]
