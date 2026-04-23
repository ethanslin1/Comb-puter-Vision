"""Split invariants: id uniqueness + zero cross-split id overlap per source.

The "live data" tests only run if the full interim parquets exist; they
are the canonical assertion the spec requires ("test_splits.py asserts
zero ID overlap across train/val/test for every source"). The
synthesized tests run always and cover edge cases directly on
``validate_source_splits``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from beevision.data.paths import load_paths
from beevision.data.splits import (
    SOURCES,
    validate_all,
    validate_source_splits,
)


# ---------- Synthesized (always run) --------------------------------------


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def test_validate_reports_ok_on_clean_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "id":    ["a", "b", "c", "d"],
        "split": ["train", "train", "val", "test"],
        "label": ["mite", "no_mite", "mite", "no_mite"],
    })
    pq = _write(df, tmp_path / "x.parquet")
    rep = validate_source_splits(pq, "x")
    assert rep.ok, rep.errors
    assert rep.n_total == 4
    assert rep.n_per_split == {"train": 2, "val": 1, "test": 1}


def test_validate_detects_id_overlap_between_splits(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "id":    ["a", "b", "a", "d"],
        "split": ["train", "train", "val", "test"],
        "label": ["mite", "mite", "mite", "no_mite"],
    })
    pq = _write(df, tmp_path / "x.parquet")
    rep = validate_source_splits(pq, "x")
    assert not rep.ok
    # 'a' is duplicated → both duplicate-id and cross-split overlap fire.
    errs = " ".join(rep.errors)
    assert "duplicate" in errs
    assert "overlap" in errs


def test_validate_detects_bad_split_values(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "id":    ["a", "b"],
        "split": ["train", "eval"],  # "eval" not allowed
        "label": ["mite", "no_mite"],
    })
    pq = _write(df, tmp_path / "x.parquet")
    rep = validate_source_splits(pq, "x")
    assert not rep.ok
    assert any("unknown split" in e for e in rep.errors)


def test_validate_handles_missing_parquet(tmp_path: Path) -> None:
    rep = validate_source_splits(tmp_path / "nope.parquet", "x")
    assert not rep.ok


# ---------- Live (skipped if full ingest hasn't been run) ------------------


def _paths_or_skip():
    """Return Paths only if every source's interim parquet is present."""
    config = Path(__file__).resolve().parent.parent / "configs" / "data.yaml"
    if not config.exists():
        pytest.skip("configs/data.yaml missing")
    paths = load_paths(config)
    missing = [s for s in SOURCES if not paths.parquet_path(s).exists()]
    if missing:
        pytest.skip(f"missing interim parquets: {missing}")
    return paths


def test_live_parquets_have_no_cross_split_id_overlap() -> None:
    paths = _paths_or_skip()
    reports = validate_all(paths)
    failures = {r.source: r.errors for r in reports if not r.ok}
    assert not failures, failures


def test_live_parquets_all_have_nonempty_splits() -> None:
    paths = _paths_or_skip()
    for rep in validate_all(paths):
        assert rep.n_total > 0, f"{rep.source}: empty"
        for split, n in rep.n_per_split.items():
            assert n > 0, f"{rep.source}/{split}: empty split"
