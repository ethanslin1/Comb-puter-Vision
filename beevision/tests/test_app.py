"""Tests for the streamlit-free app helpers (``app.data`` + ``app.components``).

We intentionally do **not** import ``app.dashboard`` — that module imports
``streamlit``, which is an optional dep. Anything that needs streamlit
gets exercised by manual ``streamlit run`` testing, not pytest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from beevision.app import (
    SCORE_BUCKETS,
    ScoreBucket,
    class_breakdown_chart_data,
    confusion_matrix_text,
    find_frame_image,
    food_breakdown_chart_data,
    format_count,
    format_pct,
    format_ratio,
    generate_demo_report,
    list_reports,
    load_report_json,
    make_demo_health_inputs,
    report_to_json,
    resolve_report_path,
    save_demo_report,
    score_bucket,
    score_components_chart_data,
    severity_for_note,
)
from beevision.metrics import HealthReport


# ---------- Demo data -----------------------------------------------------


def test_make_demo_inputs_known_qualities_round_trip() -> None:
    for q in ("healthy", "struggling", "failing"):
        cells, bees = make_demo_health_inputs(quality=q, seed=1)
        assert len(cells) > 0
        assert len(bees) > 0


def test_make_demo_inputs_unknown_quality_raises() -> None:
    with pytest.raises(ValueError, match="unknown demo quality"):
        make_demo_health_inputs(quality="bogus")


def test_demo_report_healthy_scores_high() -> None:
    report = generate_demo_report(quality="healthy", seed=1337)
    assert isinstance(report, HealthReport)
    assert report.composite_score >= 8.0
    # Sanity: all three axes should be strong on a healthy demo.
    comp = report.score_components
    assert comp["brood_health"] > 0.5
    assert comp["food_balance"] > 0.5
    assert comp["mite_safety"] > 0.8


def test_demo_report_failing_scores_low() -> None:
    report = generate_demo_report(quality="failing", seed=1337)
    assert report.composite_score <= 2.0


def test_demo_report_is_seed_deterministic() -> None:
    a = generate_demo_report(seed=42)
    b = generate_demo_report(seed=42)
    # Same numeric outputs across runs given same seed.
    assert a.composite_score == b.composite_score
    assert a.brood.regularity == pytest.approx(b.brood.regularity)


# ---------- Disk I/O ------------------------------------------------------


def test_save_and_load_demo_report_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "demo" / "report.json"
    saved = save_demo_report(out, quality="healthy", seed=99)
    assert saved.exists()
    loaded = load_report_json(saved)
    assert "composite_score" in loaded
    assert "brood" in loaded and "regularity" in loaded["brood"]


def test_load_report_accepts_directory(tmp_path: Path) -> None:
    save_demo_report(tmp_path / "report.json", quality="struggling", seed=7)
    # Pass the directory rather than the file.
    loaded = load_report_json(tmp_path)
    assert "composite_score" in loaded


def test_resolve_report_path_file_passthrough(tmp_path: Path) -> None:
    p = tmp_path / "specific.json"
    p.write_text("{}")
    assert resolve_report_path(p) == p


def test_resolve_report_path_dir_appends_filename(tmp_path: Path) -> None:
    out = resolve_report_path(tmp_path)
    assert out == tmp_path / "report.json"


def test_load_report_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_report_json(tmp_path / "does_not_exist.json")


def test_find_frame_image_picks_first_match(tmp_path: Path) -> None:
    save_demo_report(tmp_path / "report.json", seed=1)
    # No frame yet → None.
    assert find_frame_image(tmp_path) is None
    # Add a frame.png; should be found.
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    found = find_frame_image(tmp_path)
    assert found == frame


def test_list_reports_finds_subdirs_with_report(tmp_path: Path) -> None:
    # Two valid report dirs + one empty dir.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c-empty").mkdir()
    save_demo_report(tmp_path / "a" / "report.json", seed=1)
    save_demo_report(tmp_path / "b" / "report.json", seed=2)

    found = list_reports(tmp_path)
    assert {p.name for p in found} == {"a", "b"}


def test_list_reports_missing_root_returns_empty(tmp_path: Path) -> None:
    assert list_reports(tmp_path / "does_not_exist") == []


def test_report_to_json_round_trip_via_dict(tmp_path: Path) -> None:
    report = generate_demo_report(quality="healthy", seed=1)
    d = report_to_json(report)
    # Should be JSON-serializable without errors.
    raw = json.dumps(d)
    parsed = json.loads(raw)
    assert parsed["composite_score"] == report.composite_score


# ---------- Score buckets / formatters ------------------------------------


@pytest.mark.parametrize(
    "score, label",
    [
        (10.0, "excellent"),
        (9.5, "excellent"),
        (8.5, "good"),
        (6.0, "good"),
        (5.0, "warning"),
        (4.0, "warning"),
        (3.0, "critical"),
        (1.0, "critical"),
    ],
)
def test_score_bucket_correct_label(score: float, label: str) -> None:
    bucket = score_bucket(score)
    assert isinstance(bucket, ScoreBucket)
    assert bucket.label == label


def test_score_bucket_out_of_range_clamped() -> None:
    assert score_bucket(0.0).label == "critical"
    assert score_bucket(11.0).label == "excellent"


def test_score_bucket_color_is_hex() -> None:
    for label, _, _, color in SCORE_BUCKETS:
        assert color.startswith("#") and len(color) == 7


def test_format_pct_handles_none() -> None:
    assert format_pct(None) == "n/a"


def test_format_pct_default_decimals() -> None:
    assert format_pct(0.05) == "5.0%"


def test_format_ratio_none_renders_na() -> None:
    assert format_ratio(None) == "n/a"


def test_format_ratio_finite_value() -> None:
    assert format_ratio(2.5) == "2.50:1"


def test_format_count_thousands_separator() -> None:
    assert format_count(1234567) == "1,234,567"


# ---------- Chart-ready data ----------------------------------------------


def test_class_breakdown_partitions_total() -> None:
    report = report_to_json(generate_demo_report(quality="healthy", seed=1))
    rows = class_breakdown_chart_data(report)
    cats = {r["category"] for r in rows}
    assert cats == {"brood", "food", "other"}
    n_brood = report["brood"]["n_brood_cells"]
    n_food = report["composition"]["n_food_cells"]
    n_total = report["brood"]["n_total_cells"]
    counts = {r["category"]: r["count"] for r in rows}
    assert counts["brood"] == n_brood
    assert counts["food"] == n_food
    assert counts["brood"] + counts["food"] + counts["other"] == n_total


def test_food_breakdown_returns_three_food_classes() -> None:
    report = report_to_json(generate_demo_report(quality="healthy", seed=1))
    rows = food_breakdown_chart_data(report)
    assert {r["food_class"] for r in rows} == {"honey", "nectar", "pollen"}


def test_score_components_weighted_sum_close_to_raw() -> None:
    report = report_to_json(generate_demo_report(quality="healthy", seed=1))
    rows = score_components_chart_data(report)
    weighted_sum = sum(r["weighted"] for r in rows)
    raw = report["score_components"]["raw"]
    # Floating-point + rounding of the composite should match within 1e-6.
    assert weighted_sum == pytest.approx(raw, abs=1e-6)


def test_confusion_matrix_text_columns_align() -> None:
    cm = [[10, 2], [1, 7]]
    out = confusion_matrix_text(cm, classes=["no_mite", "mite"])
    lines = out.splitlines()
    # Header row and one row per class.
    assert len(lines) == 1 + len(cm)
    assert "no_mite" in lines[0] and "mite" in lines[0]


def test_confusion_matrix_class_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="class names"):
        confusion_matrix_text([[1, 0], [0, 1]], classes=["only_one"])


# ---------- Notes severity ------------------------------------------------


@pytest.mark.parametrize("note,expected", [
    ("no brood detected on this frame", "warning"),
    ("no bees detected; mite load uninformative", "warning"),
    ("no food cells detected", "warning"),
    ("everything looks fine", "info"),
])
def test_severity_for_note(note: str, expected: str) -> None:
    assert severity_for_note(note) == expected
