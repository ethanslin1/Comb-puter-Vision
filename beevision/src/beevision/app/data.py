"""Streamlit-free data layer for the dashboard.

Two responsibilities:

1. **Load HealthReport JSON from disk** — what the pipeline produces:
   ``$BEEVISION_ROOT/processed/reports/<frame_id>/report.json``. The app
   accepts either the JSON path or the containing directory; if a
   ``frame.jpg`` (or ``.png``) sits next to the JSON, ``find_frame_image``
   resolves it for display. Heatmap overlay paths follow the same pattern
   (``gradcam_<class>.png``, ``occlusion_<class>.png``).

2. **Generate a synthetic demo report** for development / when no real
   pipeline output is available yet. The demo report is a real
   ``HealthReport`` object built from synthesized cell + bee instances,
   so the score axes are computed by the actual metrics module — no
   hardcoded numbers.

Kept free of any ``streamlit`` import so unit tests can exercise this
without the optional dependency installed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from beevision.metrics import (
    BeeInstance,
    CellInstance,
    HealthReport,
    build_health_report,
)

LOG = logging.getLogger("app.data")

REPORT_FILENAME = "report.json"
FRAME_CANDIDATES = ("frame.jpg", "frame.jpeg", "frame.png")


# ---------- Disk I/O -------------------------------------------------------


def resolve_report_path(path: str | Path) -> Path:
    """Accept a file or directory; return the path to ``report.json``.

    If ``path`` is a directory, looks for ``report.json`` inside.
    """
    p = Path(path)
    if p.is_dir():
        return p / REPORT_FILENAME
    return p


def load_report_json(path: str | Path) -> dict:
    """Load a HealthReport-shaped JSON without instantiating dataclasses.

    Returning the raw dict keeps the dashboard tolerant of optional fields
    (e.g. older reports lacking ``classes_present`` or new fields the
    pipeline starts emitting). Use ``rebuild_report`` for a typed view.
    """
    json_path = resolve_report_path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"no report.json at {json_path}")
    with json_path.open("r") as fh:
        return json.load(fh)


def find_frame_image(path: str | Path) -> Path | None:
    """Look for a sibling frame image next to the report JSON.

    Returns the first match from ``FRAME_CANDIDATES`` or ``None``.
    """
    json_path = resolve_report_path(path)
    parent = json_path.parent
    for name in FRAME_CANDIDATES:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def list_reports(root: str | Path) -> list[Path]:
    """List directories under ``root`` that contain a ``report.json``.

    The pipeline writes one subdirectory per frame, so this surfaces all
    available reports for a sidebar selector.
    """
    base = Path(root)
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / REPORT_FILENAME).exists())


# ---------- Demo data ------------------------------------------------------


def _grid_cells(
    n_per_side: int,
    spacing: float,
    cls: str,
    origin: tuple[float, float] = (0.0, 0.0),
    score: float = 0.92,
) -> list[CellInstance]:
    return [
        CellInstance(
            cls=cls,
            position=(origin[0] + i * spacing, origin[1] + j * spacing),
            score=score,
        )
        for i in range(n_per_side)
        for j in range(n_per_side)
    ]


def make_demo_health_inputs(
    seed: int = 1337,
    quality: str = "healthy",
) -> tuple[list[CellInstance], list[BeeInstance]]:
    """Build synthetic cell + bee instances for one of three demo profiles.

    ``quality`` is one of ``"healthy"`` (high score), ``"struggling"``
    (mid), ``"failing"`` (low). Returned instances are deterministic given
    ``seed`` so the demo report is reproducible across reloads.
    """
    rng = np.random.default_rng(seed)

    if quality == "healthy":
        # Tight 6x6 capped-brood grid, 6 honey, 4 nectar, 2 pollen, 0 other.
        cells: list[CellInstance] = []
        cells += _grid_cells(6, spacing=1.0, cls="capped_brood", score=0.95)
        cells += [CellInstance(cls="larva", position=(7.0 + i, 0.0), score=0.88) for i in range(2)]
        cells += [CellInstance(cls="egg", position=(7.0 + i, 1.0), score=0.85) for i in range(2)]
        cells += [CellInstance(cls="honey", position=(20.0 + i, 0.0), score=0.93) for i in range(7)]
        cells += [CellInstance(cls="nectar", position=(20.0 + i, 1.0), score=0.86) for i in range(2)]
        cells += [CellInstance(cls="pollen", position=(20.0 + i, 2.0), score=0.84) for i in range(3)]
        # 12 bees with low mite signal.
        bees = [
            BeeInstance(cls="no_mite", position=(rng.uniform(0, 30), rng.uniform(0, 10)), score=float(rng.uniform(0.0, 0.15)))
            for _ in range(12)
        ]
        return cells, bees

    if quality == "struggling":
        # Sparser brood pattern (5x5, missing corners), uneven food.
        full = _grid_cells(5, spacing=1.0, cls="capped_brood", score=0.78)
        # Drop corners → "shotgun" pattern.
        keep = [c for c in full if (c.position[0], c.position[1]) not in {(0, 0), (4, 0), (0, 4), (4, 4)}]
        cells = list(keep)
        cells += [CellInstance(cls="honey", position=(20.0 + i, 0.0), score=0.81) for i in range(8)]
        cells += [CellInstance(cls="pollen", position=(20.0 + i, 1.0), score=0.78) for i in range(1)]
        # 12 bees, ~25% mite hits with moderate confidence.
        bees = []
        for i in range(12):
            is_mite = i < 3
            bees.append(BeeInstance(
                cls="mite" if is_mite else "no_mite",
                position=(rng.uniform(0, 30), rng.uniform(0, 10)),
                score=float(rng.uniform(0.55, 0.85)) if is_mite else float(rng.uniform(0.05, 0.2)),
            ))
        return cells, bees

    if quality == "failing":
        # No brood, no food, heavy mite load.
        cells = [CellInstance(cls="other", position=(float(i), 0.0), score=0.6) for i in range(6)]
        bees = [
            BeeInstance(cls="mite", position=(rng.uniform(0, 30), rng.uniform(0, 10)), score=float(rng.uniform(0.85, 0.98)))
            for _ in range(10)
        ]
        return cells, bees

    raise ValueError(f"unknown demo quality {quality!r}; choose healthy/struggling/failing")


def generate_demo_report(quality: str = "healthy", seed: int = 1337) -> HealthReport:
    """Build a real ``HealthReport`` from synthesized inputs.

    All numeric fields come from the production metrics module — the demo
    only fakes the upstream cell/bee predictions, not the math.
    """
    cells, bees = make_demo_health_inputs(seed=seed, quality=quality)
    return build_health_report(cells, bees)


def report_to_json(report: HealthReport) -> dict:
    """Serialize a HealthReport to a plain JSON dict."""
    return asdict(report)


def save_demo_report(path: str | Path, quality: str = "healthy", seed: int = 1337) -> Path:
    """Write a demo report to disk so the dashboard can load it like a real one."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    report = generate_demo_report(quality=quality, seed=seed)
    with p.open("w") as fh:
        json.dump(report_to_json(report), fh, indent=2)
    return p
