"""Streamlit-free presentation helpers.

Anything that turns a HealthReport into something renderable — score
buckets, color codes, formatted strings, chart-ready lists. Streamlit
imports happen only in ``dashboard.py``; everything here is pure Python /
NumPy and unit-testable.

Score buckets follow the README's clinical interpretation:

    9–10 : thriving        ("excellent")
    6–8  : healthy w/ minor issues  ("good")
    4–5  : at-risk; intervene       ("warning")
    1–3  : failing                  ("critical")

Colors are CSS hex; the dashboard maps these to ``st.markdown`` styling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Health-score buckets.
SCORE_BUCKETS = (
    ("excellent", 9.0, 10.0, "#16a34a"),   # green-600
    ("good",      6.0, 8.999, "#65a30d"),  # lime-600
    ("warning",   4.0, 5.999, "#eab308"),  # yellow-500
    ("critical",  1.0, 3.999, "#dc2626"),  # red-600
)


@dataclass
class ScoreBucket:
    label: str
    color: str        # CSS hex
    range_lo: float
    range_hi: float


def score_bucket(score: float) -> ScoreBucket:
    """Map a 1–10 composite score to its qualitative bucket."""
    for label, lo, hi, color in SCORE_BUCKETS:
        if lo <= score <= hi:
            return ScoreBucket(label=label, color=color, range_lo=lo, range_hi=hi)
    # Out-of-range (shouldn't happen since ``composite_score`` clamps to [1, 10])
    # but be defensive: anything below 1 maps to critical, above 10 to excellent.
    if score < 1.0:
        _, _, _, color = SCORE_BUCKETS[-1]
        return ScoreBucket(label="critical", color=color, range_lo=1.0, range_hi=3.999)
    _, _, _, color = SCORE_BUCKETS[0]
    return ScoreBucket(label="excellent", color=color, range_lo=9.0, range_hi=10.0)


# ---------- Formatters ---------------------------------------------------


def format_pct(x: float | None, decimals: int = 1) -> str:
    """Format a [0, 1] number as ``XX.X%``; ``None`` renders as 'n/a'."""
    if x is None:
        return "n/a"
    return f"{x * 100:.{decimals}f}%"


def format_ratio(x: float | None, decimals: int = 2) -> str:
    """Format a ratio (e.g., honey:pollen) as 'X.XX:1' or 'n/a'."""
    if x is None or not _is_finite(x):
        return "n/a"
    return f"{x:.{decimals}f}:1"


def format_count(n: int) -> str:
    return f"{n:,}"


def _is_finite(x: float) -> bool:
    import math
    return math.isfinite(x)


# ---------- Chart-ready data ---------------------------------------------


def class_breakdown_chart_data(report: dict) -> list[dict[str, Any]]:
    """Return rows describing the high-level cell composition.

    Output rows: ``[{"category": str, "count": int}]`` — one per
    {brood, food, other}. Suitable for ``st.bar_chart`` after a pivot.
    """
    brood = report["brood"]
    comp = report["composition"]
    n_brood = brood["n_brood_cells"]
    n_food = comp["n_food_cells"]
    n_total = brood["n_total_cells"]
    n_other = max(n_total - n_brood - n_food, 0)
    return [
        {"category": "brood", "count": int(n_brood)},
        {"category": "food",  "count": int(n_food)},
        {"category": "other", "count": int(n_other)},
    ]


def food_breakdown_chart_data(report: dict) -> list[dict[str, Any]]:
    """Per-class food storage breakdown — honey/nectar/pollen counts."""
    c = report["composition"]
    return [
        {"food_class": "honey",  "count": int(c["n_honey"])},
        {"food_class": "nectar", "count": int(c["n_nectar"])},
        {"food_class": "pollen", "count": int(c["n_pollen"])},
    ]


def score_components_chart_data(report: dict) -> list[dict[str, Any]]:
    """Sub-axis scores in [0, 1] that fed the composite.

    Each row: ``{"axis": str, "score": float, "weight": float, "weighted": float}``.
    The dashboard renders these as a stacked bar to show how each axis
    contributed to the headline score.
    """
    comps = report.get("score_components", {})
    weights = report.get("weights", {})
    out: list[dict[str, Any]] = []
    pairs = [
        ("brood",  "brood_health", weights.get("brood", 0.0)),
        ("food",   "food_balance", weights.get("food", 0.0)),
        ("mites",  "mite_safety",  weights.get("mites", 0.0)),
    ]
    for axis, comp_key, w in pairs:
        s = float(comps.get(comp_key, 0.0))
        out.append({
            "axis": axis,
            "score": s,
            "weight": float(w),
            "weighted": float(w * s),
        })
    return out


def confusion_matrix_text(matrix: list[list[int]], classes: list[str] | None = None) -> str:
    """Render a small confusion matrix as a monospaced table string.

    Used by the model-eval inspector pane (when a ``metrics.jsonl`` is
    loaded alongside a report). Returns a string suitable for
    ``st.code``.
    """
    n = len(matrix)
    if classes is None:
        classes = [f"c{i}" for i in range(n)]
    if len(classes) != n:
        raise ValueError(f"got {n}-row matrix but {len(classes)} class names")
    width = max(max(len(c) for c in classes), 5) + 1
    header = " " * (width + 2) + "".join(f"{c:>{width}}" for c in classes)
    rows = [header]
    for label, row in zip(classes, matrix):
        rows.append(f"{label:>{width}} |" + "".join(f"{v:>{width}}" for v in row))
    return "\n".join(rows)


# ---------- Notes / warnings ---------------------------------------------


def severity_for_note(note: str) -> str:
    """Map a free-text note into one of {info, warning, error} for styling."""
    s = note.lower()
    if "no bees" in s or "no brood" in s or "no food" in s:
        return "warning"
    return "info"
