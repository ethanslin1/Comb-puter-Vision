"""Torch-facing datasets over BeeVision shards.

Two consumers share one shard format:

- ``CellClassificationDataset`` (used by deepbee_cls + mite head via
  ``MiteClassificationDataset``): returns (image_tensor, class_idx).
- ``FrameSegmentationDataset`` (deepbee_seg): returns
  (image_tensor, mask_tensor).

Both back onto the tar shards written by ``shard_writer``. For lightweight
uses (tests, notebooks) a second constructor ``from_parquet`` reads the
interim parquet + PNGs directly, bypassing webdataset.

WeightedRandomSampler for the mite head is computed here
(``build_mite_train_sampler``) since it's a train-time concern tied to the
dataset's per-row label, and we want it in one place for auditing.

This module keeps the ML framework surface small on purpose: it emits
numpy + the class-index mapping, and the training loops build tensors /
samplers / loaders. The only torch dependency is to provide a
``torch.utils.data.Dataset`` base class.
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image

from beevision.data.augment import apply as apply_transforms

try:
    import torch  # noqa: F401
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch is optional for tests
    class Dataset:  # type: ignore[no-redef]
        pass

LOG = logging.getLogger("datasets")


# ---------- Label indexing -------------------------------------------------


# Kept here so train code and sanity outputs share one source of truth.
CELL_CLASSES: tuple[str, ...] = (
    "egg",
    "larva",
    "capped_brood",
    "pollen",
    "nectar",
    "honey",
    "other",
)
MITE_CLASSES: tuple[str, ...] = ("no_mite", "mite")


def label_to_index(classes: tuple[str, ...]) -> dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


CELL_LABEL_TO_IDX = label_to_index(CELL_CLASSES)
MITE_LABEL_TO_IDX = label_to_index(MITE_CLASSES)


# ---------- Sample-from-disk iteration (parquet path) ---------------------


@dataclass
class _Item:
    id: str
    image_path: Path
    target: Any
    meta: dict[str, Any]


class _ParquetBacked(Dataset):
    """Base Dataset that reads (image, target) rows from an interim parquet.

    Subclasses override ``_load_target`` to either return a class index
    (classification) or a mask array (segmentation).
    """

    def __init__(
        self,
        parquet_path: Path,
        interim_dir: Path,
        *,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.interim = Path(interim_dir)
        self.split = split
        self.transform = transform

        df = pd.read_parquet(self.parquet_path, engine="pyarrow")
        df = df[df["split"] == split].sort_values("id", kind="mergesort").reset_index(drop=True)
        if df.empty:
            LOG.warning("%s has no rows for split=%r", self.parquet_path, split)
        self._df = df

    def __len__(self) -> int:
        return len(self._df)

    def _load_image(self, rel_path: str) -> np.ndarray:
        img = Image.open(self.interim / rel_path)
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)

    def _load_target(self, row: dict) -> Any:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df.iloc[idx].to_dict()
        image = self._load_image(row["image_path"])
        target = self._load_target(row)

        if self.transform is None:
            sample: dict[str, Any] = {"image": image, "target": target, "id": row["id"]}
            if isinstance(target, np.ndarray):
                sample["mask"] = target
            return sample

        if isinstance(target, np.ndarray):
            out = apply_transforms(self.transform, image, target)
            return {"image": out["image"], "mask": out["mask"], "id": row["id"]}
        out = apply_transforms(self.transform, image)
        return {"image": out["image"], "target": target, "id": row["id"]}


# ---------- Classification datasets ---------------------------------------


class CellClassificationDataset(_ParquetBacked):
    """Cells → class index in ``CELL_CLASSES``."""

    classes = CELL_CLASSES

    def _load_target(self, row: dict) -> int:
        label = str(row["label"])
        if label not in CELL_LABEL_TO_IDX:
            raise KeyError(f"unknown cell label: {label}")
        return CELL_LABEL_TO_IDX[label]


class MiteClassificationDataset(_ParquetBacked):
    """Bee crops → {0=no_mite, 1=mite}."""

    classes = MITE_CLASSES

    def _load_target(self, row: dict) -> int:
        label = str(row["label"])
        if label not in MITE_LABEL_TO_IDX:
            raise KeyError(f"unknown mite label: {label}")
        return MITE_LABEL_TO_IDX[label]


# ---------- Segmentation dataset ------------------------------------------


class FrameSegmentationDataset(_ParquetBacked):
    """Frames + binary masks for the UNet."""

    classes = ("background", "comb")

    def _load_target(self, row: dict) -> np.ndarray:
        mask = Image.open(self.interim / row["mask_path"])
        mask.load()
        if mask.mode != "L":
            mask = mask.convert("L")
        return np.asarray(mask, dtype=np.uint8)


# ---------- Mite sampler helper -------------------------------------------


def build_mite_train_sampler(dataset: MiteClassificationDataset) -> dict[str, Any]:
    """Return the pieces needed to build a torch WeightedRandomSampler.

    We compute per-class weights inversely proportional to class frequency
    and a parallel per-sample weight vector. Training code wraps these in
    ``torch.utils.data.WeightedRandomSampler(weights, num_samples=len(...)
    , replacement=True)``.
    """
    labels = dataset._df["label"].map(MITE_LABEL_TO_IDX).to_numpy()
    if len(labels) == 0:
        return {"class_weights": np.array([1.0, 1.0]), "sample_weights": np.array([])}
    counts = np.bincount(labels, minlength=len(MITE_CLASSES)).astype(np.float64)
    # Avoid div-by-zero; if a class is absent (shouldn't happen in train),
    # use 1.0 so presence is the only signal.
    class_weights = np.where(counts > 0, counts.sum() / (len(MITE_CLASSES) * counts), 1.0)
    sample_weights = class_weights[labels]
    return {"class_weights": class_weights, "sample_weights": sample_weights}
