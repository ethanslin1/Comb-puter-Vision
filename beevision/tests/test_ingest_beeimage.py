"""Unit tests for beevision.data.ingest_beeimage helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from beevision.data.ingest_beeimage import (
    filter_and_label,
    resolve_label,
    stratified_split,
)
from beevision.data.schema import MiteLabel


PATTERNS = [
    {"pattern": "healthy", "label": "no_mite"},
    {"pattern": "varroa", "label": "mite"},
    {"pattern": "varrao", "label": "mite"},  # dataset typo
]


# ---------- resolve_label --------------------------------------------------


def test_resolve_label_healthy() -> None:
    assert resolve_label("healthy", PATTERNS) is MiteLabel.NO_MITE


def test_resolve_label_case_insensitive() -> None:
    assert resolve_label("Healthy", PATTERNS) is MiteLabel.NO_MITE
    assert resolve_label("Varroa, Small Hive Beetles", PATTERNS) is MiteLabel.MITE


def test_resolve_label_typo_variant() -> None:
    assert resolve_label("few varrao, hive beetles", PATTERNS) is MiteLabel.MITE


def test_resolve_label_drops_unmatched() -> None:
    for h in ("ant problems", "hive being robbed", "missing queen", "", "random noise"):
        assert resolve_label(h, PATTERNS) is None, f"{h!r} should be dropped"


# ---------- filter_and_label -----------------------------------------------


def test_filter_and_label_drops_unmatched_rows_and_adds_label() -> None:
    df = pd.DataFrame(
        {
            "file": ["a.png", "b.png", "c.png", "d.png"],
            "health": ["healthy", "ant problems", "Varroa, SHB", "missing queen"],
        }
    )
    out = filter_and_label(df, PATTERNS)
    assert set(out["file"]) == {"a.png", "c.png"}
    assert set(out["label"]) == {"no_mite", "mite"}
    # Deterministic row order preserved.
    assert list(out["file"]) == ["a.png", "c.png"]


# ---------- stratified_split -----------------------------------------------


def test_stratified_split_preserves_class_ratios() -> None:
    # 80 no_mite, 20 mite → stratified 70/15/15 should keep ratio per split.
    df = pd.DataFrame(
        {
            "file": [f"{i}.png" for i in range(100)],
            "label": ["no_mite"] * 80 + ["mite"] * 20,
        }
    )
    out = stratified_split(df, train=0.70, val=0.15, test=0.15, seed=1337)
    assert set(out["split"]) == {"train", "val", "test"}
    counts = out.groupby("split").size()
    # sklearn's stratified split rounds; allow ±1 off nominal.
    assert abs(counts["train"] - 70) <= 1
    assert abs(counts["val"] - 15) <= 1
    assert abs(counts["test"] - 15) <= 1
    assert counts.sum() == 100

    # Class balance within each split (within rounding of 1).
    for split in ("train", "val", "test"):
        sub = out[out["split"] == split]
        mite_frac = (sub["label"] == "mite").mean()
        assert abs(mite_frac - 0.20) < 0.05, f"{split}: mite_frac={mite_frac}"


def test_stratified_split_deterministic_across_runs() -> None:
    df = pd.DataFrame(
        {
            "file": [f"{i}.png" for i in range(40)],
            "label": (["no_mite"] * 30) + (["mite"] * 10),
        }
    )
    a = stratified_split(df, 0.7, 0.15, 0.15, seed=1337)
    b = stratified_split(df, 0.7, 0.15, 0.15, seed=1337)
    assert list(a["split"]) == list(b["split"])


def test_stratified_split_rejects_bad_ratios() -> None:
    df = pd.DataFrame({"file": ["a.png"] * 4, "label": ["no_mite"] * 2 + ["mite"] * 2})
    with pytest.raises(ValueError):
        stratified_split(df, 0.5, 0.3, 0.3, seed=1337)
