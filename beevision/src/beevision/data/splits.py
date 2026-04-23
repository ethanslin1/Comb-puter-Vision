"""Split validation across all BeeVision sources.

Splits are **always** materialized in each source's ``interim/<source>.parquet``
(the ``split`` column). This module does not *compute* splits — it loads the
already-assigned ones from disk and asserts the invariants the rest of the
pipeline depends on:

1. Every record id is unique within its source.
2. For every source, the train/val/test partitions are pairwise disjoint on
   id (zero leakage).
3. Every row's ``split`` is one of {train, val, test}.
4. Per-source class balance is sane (no split missing a class that appears
   elsewhere in the source, since the segmentation run only has one class
   pair that's trivially present everywhere). Reported, not asserted.

The CLI prints a summary table and exits 0 on success, >0 on any violation.
It is also importable: ``validate_source_splits`` / ``validate_all`` return
structured results for test_splits.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from beevision.data.paths import Paths, load_paths

LOG = logging.getLogger("splits")

# Keep this in one place so scripts/tests stay in sync with
# configs/data.yaml sources.
SOURCES: tuple[str, ...] = (
    "varroa",
    "beeimage",
    "deepbee_cls",
    "deepbee_seg",
)

VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


# ---------- Data classes ---------------------------------------------------


@dataclass
class SourceSplitReport:
    source: str
    parquet: Path
    n_total: int = 0
    n_per_split: dict[str, int] = field(default_factory=dict)
    class_col: str | None = None
    per_split_class: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------- Validation -----------------------------------------------------


def _pick_class_column(df: pd.DataFrame) -> str | None:
    """Return 'label' if present, else None (segmentation has no row-level class)."""
    return "label" if "label" in df.columns else None


def validate_source_splits(parquet_path: Path, source: str) -> SourceSplitReport:
    """Validate split invariants for a single source's parquet.

    Never raises; collects all violations into ``SourceSplitReport.errors`` so
    the caller can see the full picture.
    """
    rep = SourceSplitReport(source=source, parquet=parquet_path)

    if not parquet_path.exists():
        rep.errors.append(f"parquet missing: {parquet_path}")
        return rep

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    rep.n_total = int(len(df))
    if rep.n_total == 0:
        rep.errors.append("empty parquet")
        return rep

    # --- required columns ---
    for col in ("id", "split"):
        if col not in df.columns:
            rep.errors.append(f"missing column: {col}")
    if rep.errors:
        return rep

    # --- id uniqueness ---
    dupes = df["id"].duplicated()
    if dupes.any():
        dup_ids = df.loc[dupes, "id"].head(5).tolist()
        rep.errors.append(f"{int(dupes.sum())} duplicate id(s); first: {dup_ids}")

    # --- split value domain ---
    bad_splits = sorted(set(df["split"]) - VALID_SPLITS)
    if bad_splits:
        rep.errors.append(f"unknown split values: {bad_splits}")

    # --- pairwise disjoint id sets across splits ---
    ids_by_split: dict[str, set[str]] = {
        s: set(df.loc[df["split"] == s, "id"]) for s in sorted(set(df["split"]) & VALID_SPLITS)
    }
    split_names = list(ids_by_split)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = ids_by_split[a] & ids_by_split[b]
            if overlap:
                sample = sorted(overlap)[:5]
                rep.errors.append(
                    f"id overlap between {a!r} and {b!r}: {len(overlap)} ids "
                    f"(first 5: {sample})"
                )

    # --- per-split counts ---
    rep.n_per_split = {s: int(len(ids)) for s, ids in ids_by_split.items()}

    # --- per-split class distribution (informational) ---
    cls_col = _pick_class_column(df)
    rep.class_col = cls_col
    if cls_col:
        counts = df.groupby(["split", cls_col]).size().unstack(fill_value=0)
        rep.per_split_class = {
            str(s): {str(c): int(v) for c, v in row.items()}
            for s, row in counts.iterrows()
        }

    return rep


def validate_all(paths: Paths, sources: Iterable[str] = SOURCES) -> list[SourceSplitReport]:
    """Validate every source that has a parquet on disk; skip (warn) others."""
    reports: list[SourceSplitReport] = []
    for src in sources:
        pq = paths.parquet_path(src)
        if not pq.exists():
            LOG.warning("%s: parquet not found at %s — skipping", src, pq)
            continue
        rep = validate_source_splits(pq, src)
        reports.append(rep)
    return reports


# ---------- Reporting ------------------------------------------------------


def format_report(rep: SourceSplitReport) -> str:
    lines: list[str] = []
    status = "OK" if rep.ok else "FAIL"
    lines.append(f"[{status}] {rep.source}  n={rep.n_total}  splits={rep.n_per_split}")
    if rep.per_split_class:
        for split, cc in rep.per_split_class.items():
            pieces = ", ".join(f"{k}={v}" for k, v in sorted(cc.items()))
            lines.append(f"    {split}: {pieces}")
    for e in rep.errors:
        lines.append(f"    ERROR: {e}")
    return "\n".join(lines)


# ---------- CLI ------------------------------------------------------------


def _setup_logging(paths: Paths) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "splits.log"),
        ],
        force=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate split invariants across all sources.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip writing the JSON summary (still prints to stdout).")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Optional path for a machine-readable JSON summary.")
    args = ap.parse_args()

    paths = load_paths(args.config)
    _setup_logging(paths)

    reports = validate_all(paths)
    for r in reports:
        print(format_report(r))
    summary = {
        r.source: {
            "ok": r.ok,
            "n_total": r.n_total,
            "n_per_split": r.n_per_split,
            "errors": r.errors,
            "per_split_class": r.per_split_class,
        }
        for r in reports
    }

    out_path = args.json_out or (paths.sanity / "splits_summary.json")
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        LOG.info("summary → %s", out_path)

    any_fail = any(not r.ok for r in reports)
    return 0 if (reports and not any_fail) else (1 if not reports else 2)


if __name__ == "__main__":
    raise SystemExit(main())
