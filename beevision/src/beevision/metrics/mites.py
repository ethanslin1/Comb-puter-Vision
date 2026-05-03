"""Mite-load metrics from per-bee predictions.

Two flavors:

- **Hard mite load** = ``n_mite / n_bees`` based on the classifier's argmax
  decisions. Standard, easy to interpret, but discards confidence — a borderline
  bee at p(mite)=0.51 counts the same as a confident p(mite)=0.99.
- **Soft (probability-weighted) mite load** = mean of the positive-class
  probability across all bees. More robust under classifier miscalibration;
  borderline detections contribute partial credit.

We report both. The composite-score module uses the soft version as the
default penalty signal but exposes the hard count for human-readable output.

Empty inputs: ``mite_load`` and ``mite_load_weighted`` are 0 (treated as
"no infestation observed" rather than NaN). Calling code should check
``n_bees == 0`` before drawing conclusions — a frame with no detectable
bees is uninformative, not healthy.
"""
from __future__ import annotations

from beevision.metrics.types import BeeInstance, MiteMetrics


def compute_mite_metrics(bees: list[BeeInstance]) -> MiteMetrics:
    """Compute hard + soft mite-load from per-bee predictions."""
    n_bees = len(bees)
    if n_bees == 0:
        return MiteMetrics(
            n_bees=0,
            n_mite=0,
            mite_load=0.0,
            mite_load_weighted=0.0,
        )

    n_mite = sum(1 for b in bees if b.cls == "mite")

    # Soft load: average positive-class probability.
    # If a bee has no score, fall back to its hard label (1.0 if mite else 0.0).
    pos_probs = []
    for b in bees:
        if b.score is not None:
            pos_probs.append(b.score)
        else:
            pos_probs.append(1.0 if b.cls == "mite" else 0.0)

    return MiteMetrics(
        n_bees=n_bees,
        n_mite=n_mite,
        mite_load=float(n_mite / n_bees),
        mite_load_weighted=float(sum(pos_probs) / n_bees),
    )
