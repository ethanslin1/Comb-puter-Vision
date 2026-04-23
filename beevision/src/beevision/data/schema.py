"""Unified record schemas for BeeVision datasets.

Three record types, one per training task:

- CellRecord  — single comb cell, used for cell-type classification heads and
  as supervision for the segmentation UNet (deepbee_cls, beeimage).
- FrameRecord — full comb frame + pixel-wise class-index mask (deepbee_seg).
- MiteRecord  — single bee crop labelled mite vs. no_mite (varroa, beeimage).

All records are plain pydantic v2 models. They serialize losslessly to Parquet
via the ``records_to_dataframe`` / ``dataframe_to_records`` helpers; each source
writes one Parquet file under ``interim/``.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Type, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------- Enums -----------------------------------------------------------


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CellLabel(str, Enum):
    EGG = "egg"
    LARVA = "larva"
    CAPPED_BROOD = "capped_brood"
    POLLEN = "pollen"
    NECTAR = "nectar"
    HONEY = "honey"
    OTHER = "other"


class MiteLabel(str, Enum):
    MITE = "mite"
    NO_MITE = "no_mite"


class CellSource(str, Enum):
    DEEPBEE_CLS = "deepbee_cls"
    BEEIMAGE = "beeimage"


class FrameSource(str, Enum):
    DEEPBEE_SEG = "deepbee_seg"


class MiteSource(str, Enum):
    VARROA = "varroa"
    BEEIMAGE = "beeimage"


# ---------- Records ---------------------------------------------------------


class _BaseRecord(BaseModel):
    """Shared fields / config for every record type."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    id: str = Field(..., description="Globally unique record id; stable across reruns.")
    image_path: Path = Field(..., description="Path (relative to BEEVISION_ROOT) of the normalized image.")
    split: Split
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("id must be non-empty")
        return v


class CellRecord(_BaseRecord):
    source: CellSource
    label: CellLabel


class FrameRecord(_BaseRecord):
    source: FrameSource = FrameSource.DEEPBEE_SEG
    mask_path: Path = Field(..., description="Single-channel class-index PNG (0..N-1).")


class MiteRecord(_BaseRecord):
    source: MiteSource
    label: MiteLabel


Record = TypeVar("Record", CellRecord, FrameRecord, MiteRecord)


# ---------- Serialization helpers -------------------------------------------


def _record_to_row(rec: BaseModel) -> dict[str, Any]:
    """Flatten a record to a Parquet-friendly dict (paths → str, enums → value)."""
    d = rec.model_dump(mode="json")
    for k in ("image_path", "mask_path"):
        if k in d and d[k] is not None:
            d[k] = str(d[k])
    return d


def records_to_dataframe(records: Sequence[BaseModel]) -> pd.DataFrame:
    """Serialize a homogeneous sequence of records to a DataFrame.

    Raises ValueError if the sequence mixes record types or is empty.
    """
    records = list(records)
    if not records:
        raise ValueError("records_to_dataframe: cannot serialize empty sequence (ambiguous schema)")
    types = {type(r) for r in records}
    if len(types) != 1:
        raise ValueError(f"records_to_dataframe: mixed types {types}")
    return pd.DataFrame([_record_to_row(r) for r in records])


def dataframe_to_records(df: pd.DataFrame, model: Type[Record]) -> List[Record]:
    """Reconstruct records from a DataFrame. Validates every row through the model."""
    rows: Iterable[dict[str, Any]] = df.to_dict(orient="records")
    return [model.model_validate(r) for r in rows]


def write_parquet(records: Sequence[BaseModel], path: Path) -> Path:
    """Serialize records to Parquet. Writes deterministically (sorted by id)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records).sort_values("id", kind="mergesort").reset_index(drop=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return path


def read_parquet(path: Path, model: Type[Record]) -> List[Record]:
    df = pd.read_parquet(path, engine="pyarrow")
    return dataframe_to_records(df, model)


__all__ = [
    "Split",
    "CellLabel",
    "MiteLabel",
    "CellSource",
    "FrameSource",
    "MiteSource",
    "CellRecord",
    "FrameRecord",
    "MiteRecord",
    "records_to_dataframe",
    "dataframe_to_records",
    "write_parquet",
    "read_parquet",
]
