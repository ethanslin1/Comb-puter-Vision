"""Tests for src/beevision/data/schema.py — record types + parquet round-trip."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from beevision.data.schema import (
    CellLabel,
    CellRecord,
    CellSource,
    FrameRecord,
    FrameSource,
    MiteLabel,
    MiteRecord,
    MiteSource,
    Split,
    dataframe_to_records,
    read_parquet,
    records_to_dataframe,
    write_parquet,
)


# ---------- Cell records ----------------------------------------------------


def test_cell_record_accepts_valid_inputs() -> None:
    rec = CellRecord(
        id="deepbee_cls:0",
        source=CellSource.DEEPBEE_CLS,
        image_path=Path("interim/images/deepbee_cls/0.png"),
        label=CellLabel.CAPPED_BROOD,
        split=Split.TRAIN,
        meta={"orig_image": "FILE0646.JPG", "x": 1950, "y": 1333},
    )
    assert rec.label is CellLabel.CAPPED_BROOD
    assert rec.split is Split.TRAIN


def test_cell_record_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        CellRecord(
            id="x",
            source=CellSource.DEEPBEE_CLS,
            image_path=Path("a.png"),
            label="nonsense",  # type: ignore[arg-type]
            split=Split.TRAIN,
        )


def test_cell_record_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        CellRecord(
            id="x",
            source="varroa",  # not a cell source
            image_path=Path("a.png"),
            label=CellLabel.EGG,
            split=Split.TRAIN,
        )


def test_cell_record_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        CellRecord(
            id="",
            source=CellSource.BEEIMAGE,
            image_path=Path("a.png"),
            label=CellLabel.OTHER,
            split=Split.VAL,
        )


# ---------- Frame records ---------------------------------------------------


def test_frame_record_requires_mask_path() -> None:
    with pytest.raises(ValidationError):
        FrameRecord(  # type: ignore[call-arg]
            id="f1",
            image_path=Path("frames/f1.png"),
            split=Split.TRAIN,
        )


def test_frame_record_defaults_source() -> None:
    rec = FrameRecord(
        id="f1",
        image_path=Path("frames/f1.png"),
        mask_path=Path("masks/f1.png"),
        split=Split.VAL,
    )
    assert rec.source is FrameSource.DEEPBEE_SEG


# ---------- Mite records ----------------------------------------------------


def test_mite_record_both_sources_valid() -> None:
    for src in (MiteSource.VARROA, MiteSource.BEEIMAGE):
        rec = MiteRecord(
            id=f"{src.value}:1",
            source=src,
            image_path=Path(f"interim/images/{src.value}/1.png"),
            label=MiteLabel.MITE,
            split=Split.TEST,
        )
        assert rec.label is MiteLabel.MITE


# ---------- Parquet round-trip ---------------------------------------------


def _make_cell_records(n: int = 3) -> list[CellRecord]:
    return [
        CellRecord(
            id=f"c{i:03d}",
            source=CellSource.DEEPBEE_CLS,
            image_path=Path(f"interim/images/deepbee_cls/{i}.png"),
            label=list(CellLabel)[i % len(CellLabel)],
            split=Split.TRAIN,
            meta={"idx": i},
        )
        for i in range(n)
    ]


def test_records_to_dataframe_serializes_enums_and_paths() -> None:
    records = _make_cell_records(2)
    df = records_to_dataframe(records)
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["source"] == "deepbee_cls"
    assert df.iloc[0]["split"] == "train"
    assert isinstance(df.iloc[0]["image_path"], str)


def test_records_to_dataframe_rejects_empty() -> None:
    with pytest.raises(ValueError):
        records_to_dataframe([])


def test_records_to_dataframe_rejects_mixed_types() -> None:
    mixed = [
        _make_cell_records(1)[0],
        MiteRecord(
            id="m",
            source=MiteSource.VARROA,
            image_path=Path("a.png"),
            label=MiteLabel.NO_MITE,
            split=Split.TRAIN,
        ),
    ]
    with pytest.raises(ValueError):
        records_to_dataframe(mixed)


def test_parquet_round_trip_is_lossless(tmp_path: Path) -> None:
    records = _make_cell_records(5)
    out = write_parquet(records, tmp_path / "deepbee_cls.parquet")
    assert out.exists()

    restored = read_parquet(out, CellRecord)
    assert len(restored) == len(records)

    by_id = {r.id: r for r in records}
    for r in restored:
        orig = by_id[r.id]
        assert r.source == orig.source
        assert r.label == orig.label
        assert r.split == orig.split
        assert str(r.image_path) == str(orig.image_path)
        assert r.meta == orig.meta


def test_write_parquet_is_deterministic(tmp_path: Path) -> None:
    records = list(reversed(_make_cell_records(4)))  # write in arbitrary order
    a = write_parquet(records, tmp_path / "a.parquet")
    b = write_parquet(records, tmp_path / "b.parquet")
    assert a.read_bytes() == b.read_bytes()


def test_dataframe_to_records_validates_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "id": "c1",
                "source": "deepbee_cls",
                "image_path": "x.png",
                "label": "INVALID",
                "split": "train",
                "meta": {},
            }
        ]
    )
    with pytest.raises(Exception):  # pydantic ValidationError
        dataframe_to_records(df, CellRecord)
