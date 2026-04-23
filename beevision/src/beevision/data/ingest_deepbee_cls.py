"""Ingest DS-COMB-PT (DeepBee classification) → interim/deepbee_cls.parquet.

This is the fussiest source. The DeepBee release ships:

1. ``images_urls.csv``  — 2206 full-resolution BEE_HOPE frames + URLs
   (no header). Serves only as the universe of known frames.
2. ``labels_train.csv`` — headerless; columns (id, x, y, class_pt, image).
   ``class_pt`` is a Portuguese-ish token (eggs, larves, capped, pollen,
   nectar, honey, dontcare). ``dontcare`` is the "garbage" class; drop it.
3. ``labels_test.csv``  — has a header row with columns
   (id, x, y, radius, class (int), ``class name`` (English), ``image name``).
   We prefer ``class name``; ``class`` (int) is ignored.

Several quirks the spec calls out:

* Labels are Portuguese → translate via ``label_map`` in config.
* ``labels_test.csv`` references 13 images; only BEE_HOPE frames from DS-COMB
  are present on disk here (10 DSC_* frames live in a separate collection we
  never downloaded). Cells whose image is missing are dropped with a loud
  warning.
* Train/test images can overlap (3 BEE_HOPE frames are in both). Test wins:
  any train row whose image appears in test is dropped.
* Carve a stratified val slice (``val_fraction``, default 15%) out of what's
  left in train, using the unified English label as the stratum.

For each cell we cut a ``crop=224`` square centered on (x, y). If the cell is
close to the frame edge the crop box is shifted so it stays inside the frame
(no black padding at preprocess time; reflection-padding happens at train
time per spec). Frames are decoded ONCE and all of their cells cropped before
releasing the frame — 6000×4000 JPGs are ~13 MB each, so per-cell decoding is
a non-starter.

Outputs::

    interim/images/deepbee_cls/<split>/<stem>_<cell_id>.png   # 224 RGB
    interim/deepbee_cls.parquet                               # CellRecord
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from beevision.data.paths import Paths, load_paths
from beevision.data.schema import (
    CellLabel,
    CellRecord,
    CellSource,
    Split,
    write_parquet,
)

SOURCE = "deepbee_cls"
LOG = logging.getLogger(f"ingest_{SOURCE}")

TRAIN_COLS = ["cell_id", "x", "y", "class_pt", "image"]


# ---------- Logging ---------------------------------------------------------


def setup_logging(paths: Paths, dry_run: bool) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if not dry_run:
        handlers.append(logging.FileHandler(paths.logs / f"ingest_{SOURCE}.log"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


# ---------- Label resolution ------------------------------------------------


def normalize_label(raw: str, label_map: dict[str, str]) -> str | None:
    """Map a raw PT/EN label string to unified English, or None to drop.

    Case-insensitive lookup; unknown labels return None (caller decides).
    """
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    return label_map.get(key)


def load_train(csv_path: Path, label_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=TRAIN_COLS)
    df["label"] = df["class_pt"].apply(lambda s: normalize_label(s, label_map))
    unknown = df.loc[df["label"].isna() & (df["class_pt"].str.lower() != "dontcare"), "class_pt"]
    if len(unknown):
        LOG.warning(
            "train: %d rows with unmapped classes %s — dropping",
            len(unknown),
            sorted(set(unknown))[:10],
        )
    before = len(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    LOG.info("train: %d → %d rows after label mapping (dontcare + unmapped dropped)", before, len(df))
    return df


def load_test(csv_path: Path, label_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"image name": "image", "class name": "class_en", "id": "cell_id"})
    df["label"] = df["class_en"].apply(lambda s: normalize_label(s, label_map))
    unknown = df.loc[df["label"].isna(), "class_en"]
    if len(unknown):
        LOG.warning("test: %d rows unmapped classes %s", len(unknown), sorted(set(unknown))[:10])
    before = len(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    LOG.info("test: %d → %d rows after label mapping", before, len(df))
    return df


# ---------- Reconcile train / test / disk ----------------------------------


def reconcile(
    train: pd.DataFrame, test: pd.DataFrame, on_disk: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Enforce spec rules: test wins on overlap; drop rows missing on disk."""
    stats: dict = {}

    # Test: drop rows whose image is not on disk.
    test_present = test[test["image"].isin(on_disk)].reset_index(drop=True)
    test_missing = test[~test["image"].isin(on_disk)]
    stats["test_rows_missing_image"] = int(len(test_missing))
    stats["test_missing_unique_images"] = sorted(set(test_missing["image"]))
    if len(test_missing):
        LOG.warning(
            "test: dropped %d rows across %d missing images (first 5: %s)",
            len(test_missing),
            test_missing["image"].nunique(),
            sorted(set(test_missing["image"]))[:5],
        )

    # Train: drop rows whose image is in test (test wins), AND rows missing on disk.
    test_images = set(test_present["image"])
    t_before = len(train)
    train = train[~train["image"].isin(test_images)]
    stats["train_rows_dropped_test_overlap"] = t_before - len(train)
    t_before = len(train)
    train = train[train["image"].isin(on_disk)].reset_index(drop=True)
    stats["train_rows_missing_image"] = t_before - len(train)
    return train, test_present, stats


# ---------- Val carve -------------------------------------------------------


def carve_val(train: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """Assign train/val to each row with a stratified split on label."""
    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction must be in (0,1); got {val_fraction}")

    idx = train.index.to_numpy()
    y = train["label"].to_numpy()
    idx_train, idx_val, *_ = train_test_split(
        idx, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    splits = pd.Series(index=train.index, dtype=object)
    splits.loc[idx_train] = "train"
    splits.loc[idx_val] = "val"
    out = train.copy()
    out["split"] = splits.to_numpy()
    return out


# ---------- Cropping --------------------------------------------------------


def center_crop(img: Image.Image, x: int, y: int, side: int) -> Image.Image:
    """Return a ``side×side`` crop centered at (x,y), shifted to stay in bounds.

    This is a lossless geometric fit — we never pad here. Training will add
    reflection padding as needed (spec non-negotiable #1).
    """
    W, H = img.size
    half = side // 2
    left = max(0, min(W - side, x - half))
    top = max(0, min(H - side, y - half))
    # If frame is smaller than side on either axis, resize up preserving aspect
    # so we always emit side×side. (Not expected for DS-COMB frames.)
    if W < side or H < side:
        scale = side / min(W, H)
        img = img.resize((int(round(W * scale)), int(round(H * scale))), Image.Resampling.LANCZOS)
        return center_crop(img, int(round(x * scale)), int(round(y * scale)), side)
    return img.crop((left, top, left + side, top + side))


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


# ---------- Hashing ---------------------------------------------------------


def hash_config(
    train_csv: Path,
    test_csv: Path,
    label_map: dict,
    val_fraction: float,
    seed: int,
    crop: int,
) -> str:
    h = hashlib.sha256()
    h.update(train_csv.read_bytes())
    h.update(test_csv.read_bytes())
    h.update(str(sorted(label_map.items())).encode())
    h.update(f"|val={val_fraction}|seed={seed}|crop={crop}".encode())
    return h.hexdigest()


def sentinel_path(paths: Paths) -> Path:
    paths.cache.mkdir(parents=True, exist_ok=True)
    return paths.cache / f"{SOURCE}.ingest.hash"


# ---------- Core ingest -----------------------------------------------------


def ingest(paths: Paths, dry_run: bool = False, limit: int | None = None) -> int:
    paths.ensure()
    raw_dir = paths.raw_source_dir(SOURCE)
    src_cfg = paths.sources[SOURCE]

    train_csv = raw_dir / src_cfg["labels_train"]
    test_csv = raw_dir / src_cfg["labels_test"]
    for p in (train_csv, test_csv):
        if not p.exists():
            LOG.error("missing required CSV: %s", p)
            return 2

    label_map: dict[str, str] = dict(src_cfg["label_map"])
    val_fraction: float = float(src_cfg.get("val_fraction", 0.15))
    seed = int(paths.cfg["seed"])
    crop = int(paths.cfg["image_sizes"]["crop"])

    current_hash = hash_config(train_csv, test_csv, label_map, val_fraction, seed, crop)
    out_parquet = paths.parquet_path(SOURCE)
    sentinel = sentinel_path(paths)
    if (
        not dry_run
        and limit is None
        and sentinel.exists()
        and sentinel.read_text().strip() == current_hash
        and out_parquet.exists()
    ):
        LOG.info("up to date (hash=%s…); skipping", current_hash[:12])
        return 0

    # Enumerate images on disk (filename → absolute path). The raw dir holds
    # 2206 JPGs alongside the 3 CSVs, so filter by extension.
    on_disk_files = {p.name: p for p in raw_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    on_disk = set(on_disk_files.keys())
    LOG.info("images on disk: %d", len(on_disk))

    train = load_train(train_csv, label_map)
    test = load_test(test_csv, label_map)
    train, test, recon_stats = reconcile(train, test, on_disk)
    LOG.info("reconcile stats: %s", recon_stats)

    train = train.sort_values(["image", "cell_id"], kind="mergesort").reset_index(drop=True)
    test = test.sort_values(["image", "cell_id"], kind="mergesort").reset_index(drop=True)

    train_val = carve_val(train, val_fraction=val_fraction, seed=seed)
    test = test.copy()
    test["split"] = "test"

    # Common columns only; concat and keep deterministic order.
    cols = ["cell_id", "x", "y", "label", "image", "split"]
    # Attach radius as nullable metadata (only test has it).
    if "radius" not in train_val.columns:
        train_val["radius"] = pd.NA
    if "radius" not in test.columns:
        test["radius"] = pd.NA
    all_rows = pd.concat(
        [train_val[cols + ["radius"]], test[cols + ["radius"]]], ignore_index=True
    )

    counts = all_rows.groupby(["split", "label"]).size()
    LOG.info("split × label counts:\n%s", counts.to_string())

    if limit is not None:
        all_rows = all_rows.head(limit).reset_index(drop=True)
    LOG.info("processing %d rows (limit=%s)", len(all_rows), limit)

    if dry_run:
        for _, row in all_rows.head(5).iterrows():
            LOG.info(
                "DRY image=%s cell=%s (x,y)=(%s,%s) label=%s split=%s",
                row["image"], row["cell_id"], row["x"], row["y"], row["label"], row["split"],
            )
        LOG.info("dry-run: would write %d records → %s", len(all_rows), out_parquet)
        return 0

    paths.interim_images_dir(SOURCE).mkdir(parents=True, exist_ok=True)

    # Decode each frame once; crop all its cells.
    records: list[CellRecord] = []
    failures = 0
    per_image = all_rows.groupby("image", sort=True)

    for image_name, group in tqdm(per_image, total=per_image.ngroups, desc=SOURCE, unit="img"):
        src_path = on_disk_files.get(image_name)
        if src_path is None:
            LOG.warning("missing frame on disk at crop-time: %s", image_name)
            failures += len(group)
            continue
        try:
            img = Image.open(src_path)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:  # noqa: BLE001
            LOG.warning("decode failed %s: %s", image_name, e)
            failures += len(group)
            continue

        frame_stem = Path(image_name).stem
        for _, row in group.iterrows():
            try:
                x = int(row["x"])
                y = int(row["y"])
                split_str = str(row["split"])
                crop_img = center_crop(img, x, y, crop)

                cell_id = int(row["cell_id"])
                rec_id = f"{SOURCE}:{frame_stem}:{cell_id:07d}"
                out_rel = Path("images") / SOURCE / split_str / f"{frame_stem}__{cell_id:07d}.png"
                out_abs = paths.interim / out_rel
                out_abs.parent.mkdir(parents=True, exist_ok=True)
                out_abs.write_bytes(encode_png(crop_img))

                radius = row.get("radius", pd.NA)
                meta = {
                    "orig_image": image_name,
                    "orig_cell_id": cell_id,
                    "x": x,
                    "y": y,
                    "radius": None if pd.isna(radius) else int(radius),
                }
                records.append(
                    CellRecord(
                        id=rec_id,
                        source=CellSource.DEEPBEE_CLS,
                        image_path=out_rel,
                        label=CellLabel(row["label"]),
                        split=Split(split_str),
                        meta=meta,
                    )
                )
            except Exception as e:  # noqa: BLE001
                failures += 1
                LOG.warning("skip %s cell %s: %s", image_name, row.get("cell_id"), e)

        img.close()

    write_parquet(records, out_parquet)
    if limit is None:
        sentinel.write_text(current_hash)

    LOG.info(
        "wrote %d records → %s (failures=%d)", len(records), out_parquet, failures
    )
    return 0


# ---------- CLI -------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest DS-COMB-PT classification into interim/.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    paths = load_paths(args.config)
    setup_logging(paths, args.dry_run)
    return ingest(paths, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())


# Re-exports for testability of unused helper names is unnecessary;
# ``_DONTCARE_TOKEN`` and similar constants intentionally live in the config.
_ = Iterable  # keep import used if typing runtime eval changes
_ = Counter
