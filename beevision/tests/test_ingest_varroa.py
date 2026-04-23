"""Unit tests for beevision.data.ingest_varroa helpers."""
from __future__ import annotations

import pytest
from PIL import Image

from beevision.data.ingest_varroa import (
    infer_split,
    make_record_id,
    parse_gt_line,
    read_gt_csv,
    resize_short_edge,
)
from beevision.data.schema import Split


# ---------- gt.csv parsing --------------------------------------------------


def test_parse_gt_line_no_mite() -> None:
    path, count, boxes = parse_gt_line(
        "test/videos/a/b.mp4-bee_id_1-15-1.png 0\n"
    )
    assert path == "test/videos/a/b.mp4-bee_id_1-15-1.png"
    assert count == 0
    assert boxes == []


def test_parse_gt_line_single_mite_single_box() -> None:
    path, count, boxes = parse_gt_line("train/x.png 1 10 20 30 40")
    assert path == "train/x.png"
    assert count == 1
    assert boxes == [(10, 20, 30, 40)]


def test_parse_gt_line_multiple_boxes() -> None:
    _, count, boxes = parse_gt_line("val/x.png 2 1 2 3 4 5 6 7 8")
    assert count == 2
    assert boxes == [(1, 2, 3, 4), (5, 6, 7, 8)]


def test_parse_gt_line_rejects_malformed_bbox_coords() -> None:
    with pytest.raises(ValueError):
        parse_gt_line("val/x.png 1 1 2 3")  # only 3 bbox values


def test_parse_gt_line_rejects_missing_count() -> None:
    with pytest.raises(ValueError):
        parse_gt_line("val/x.png")


# ---------- split inference -------------------------------------------------


def test_infer_split_maps_all_three() -> None:
    assert infer_split("train/foo.png") is Split.TRAIN
    assert infer_split("val/foo.png") is Split.VAL
    assert infer_split("test/foo.png") is Split.TEST


def test_infer_split_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        infer_split("other/foo.png")


# ---------- resize ---------------------------------------------------------


def test_resize_short_edge_preserves_aspect() -> None:
    img = Image.new("RGB", (600, 300), color=(0, 0, 0))  # w=600 h=300
    out = resize_short_edge(img, 100)
    assert min(out.size) == 100
    # aspect ratio preserved
    assert abs((out.size[0] / out.size[1]) - 2.0) < 1e-6


def test_resize_short_edge_is_idempotent_when_equal() -> None:
    img = Image.new("RGB", (100, 100))
    out = resize_short_edge(img, 100)
    assert out.size == (100, 100)
    assert out is img  # no-op


# ---------- id stability ---------------------------------------------------


def test_make_record_id_stable_and_slashless() -> None:
    a = make_record_id("train/videos/x/y.png")
    b = make_record_id("train/videos/x/y.png")
    assert a == b
    assert "/" not in a
    assert a.startswith("varroa:")


# ---------- read_gt_csv roundtrip via tmp file -----------------------------


def test_read_gt_csv_sorts_and_skips_blanks(tmp_path) -> None:
    p = tmp_path / "gt.csv"
    p.write_text(
        "val/b.png 0\n"
        "train/a.png 1 10 10 20 20\n"
        "\n"
        "test/c.png 0\n"
    )
    rows = read_gt_csv(p)
    paths = [r[0] for r in rows]
    assert paths == sorted(paths)
    assert len(rows) == 3
