"""Ingest BeeImage (Kaggle) → interim/beeimage.parquet + PNGs.

The BeeImage dataset ships as ``bee_data.csv`` + ``bee_imgs/bee_imgs/*.png``.
Only the Varroa-relevant subset is useful here: rows whose ``health`` column
matches a kept pattern (``healthy`` or ``varroa``/``varrao``) are mapped to a
binary MiteLabel; every other class (ants, hive beetles w/o varroa, missing
queen, hive being robbed) is dropped per the pipeline spec.

A fresh, reproducible 70/15/15 stratified split is produced (seed=1337).

Outputs::

    interim/images/beeimage/<split>/<id>.png    # 224-short-edge RGB
    interim/beeimage.parquet                    # MiteRecord rows
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from beevision.data.paths import Paths, load_paths
from beevision.data.schema import (
    MiteLabel,
    MiteRecord,
    MiteSource,
    Split,
    write_parquet,
)

SOURCE = "beeimage"
LOG = logging.getLogger(f"ingest_{SOURCE}")


# ---------- Logging ---------------------------------------------------------


def setup_logging(paths: Paths, dry_run: bool) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if not dry_run:
        handlers.append(logging.FileHandler(paths.logs / f"ingest_{SOURCE}.log"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


# ---------- Label resolution ------------------------------------------------


def resolve_label(health: str, patterns: list[dict]) -> MiteLabel | None:
    """Return the MiteLabel for a row, or ``None`` if it should be dropped.

    Case-insensitive substring match. First pattern wins; keep the ordering
    deterministic (config order).
    """
    h = (health or "").strip().lower()
    for entry in patterns:
        if entry["pattern"].lower() in h:
            return MiteLabel(entry["label"])
    return None


def filter_and_label(
    df: pd.DataFrame, patterns: list[dict]
) -> pd.DataFrame:
    """Return a new frame with a 'label' column; unlabeled rows dropped."""
    labels = df["health"].apply(lambda h: resolve_label(h, patterns))
    keep = labels.notna()
    out = df.loc[keep].copy()
    out["label"] = labels[keep].map(lambda ml: ml.value)
    return out.reset_index(drop=True)


# ---------- Splitting -------------------------------------------------------


def stratified_split(
    df: pd.DataFrame, train: float, val: float, test: float, seed: int
) -> pd.DataFrame:
    """Assign a deterministic 'split' column: train/val/test by label strata."""
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError(f"splits must sum to 1.0, got {train + val + test}")

    idx = df.index.to_numpy()
    y = df["label"].to_numpy()

    idx_train, idx_rest, y_train, y_rest = train_test_split(
        idx, y, test_size=(1 - train), stratify=y, random_state=seed
    )
    rel_val = val / (val + test)
    idx_val, idx_test, *_ = train_test_split(
        idx_rest, y_rest, test_size=(1 - rel_val), stratify=y_rest, random_state=seed
    )

    splits = pd.Series(index=df.index, dtype=object)
    splits.loc[idx_train] = "train"
    splits.loc[idx_val] = "val"
    splits.loc[idx_test] = "test"
    out = df.copy()
    out["split"] = splits.to_numpy()
    return out


# ---------- Image handling --------------------------------------------------


def resize_short_edge(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    if s == short:
        return img
    scale = short / float(s)
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


# ---------- Hashing ---------------------------------------------------------


def hash_config(csv_path: Path, patterns: list[dict], splits_cfg: dict, seed: int, crop: int) -> str:
    h = hashlib.sha256()
    h.update(csv_path.read_bytes())
    h.update(str(patterns).encode())
    h.update(str(sorted(splits_cfg.items())).encode())
    h.update(f"|seed={seed}|crop={crop}".encode())
    return h.hexdigest()


def sentinel_path(paths: Paths) -> Path:
    paths.cache.mkdir(parents=True, exist_ok=True)
    return paths.cache / f"{SOURCE}.ingest.hash"


# ---------- Core ingest -----------------------------------------------------


def ingest(paths: Paths, dry_run: bool = False, limit: int | None = None) -> int:
    paths.ensure()
    raw_dir = paths.raw_source_dir(SOURCE)
    src_cfg = paths.sources[SOURCE]

    csv_path = raw_dir / src_cfg["labels_file"]
    images_dir = raw_dir / src_cfg["images_dir"]
    if not csv_path.exists():
        LOG.error("CSV missing: %s", csv_path)
        return 2
    if not images_dir.exists():
        LOG.error("images dir missing: %s", images_dir)
        return 2

    patterns = src_cfg["health_keep_patterns"]
    splits_cfg = src_cfg["splits"]
    seed = int(paths.cfg["seed"])
    crop = int(paths.cfg["image_sizes"]["crop"])

    current_hash = hash_config(csv_path, patterns, splits_cfg, seed, crop)
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

    LOG.info("reading %s", csv_path)
    df = pd.read_csv(csv_path)
    LOG.info("raw rows: %d", len(df))

    labeled = filter_and_label(df, patterns)
    LOG.info("kept %d / %d rows after health filter", len(labeled), len(df))
    if len(labeled) == 0:
        LOG.error("no rows left after filter — check health_keep_patterns")
        return 2

    labeled = labeled.sort_values("file", kind="mergesort").reset_index(drop=True)
    labeled = stratified_split(
        labeled,
        train=splits_cfg["train"],
        val=splits_cfg["val"],
        test=splits_cfg["test"],
        seed=seed,
    )

    if limit is not None:
        labeled = labeled.head(limit).reset_index(drop=True)
    LOG.info("processing %d rows (limit=%s)", len(labeled), limit)

    counts = labeled.groupby(["split", "label"]).size()
    LOG.info("split × label counts:\n%s", counts.to_string())

    if dry_run:
        for _, row in labeled.head(5).iterrows():
            LOG.info(
                "DRY file=%s health=%r label=%s split=%s",
                row["file"], row["health"], row["label"], row["split"],
            )
        LOG.info("dry-run: would write %d records → %s", len(labeled), out_parquet)
        return 0

    out_root = paths.interim_images_dir(SOURCE)
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[MiteRecord] = []
    failures = 0
    for _, row in tqdm(list(labeled.iterrows()), desc=SOURCE, unit="img"):
        file_name: str = row["file"]
        split_str: str = row["split"]
        label_str: str = row["label"]
        src_path = images_dir / file_name
        if not src_path.exists():
            LOG.warning("missing image on disk: %s", src_path)
            failures += 1
            continue
        try:
            img = Image.open(src_path)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = resize_short_edge(img, crop)

            stem = Path(file_name).stem
            rec_id = f"{SOURCE}:{stem}"
            out_rel = Path("images") / SOURCE / split_str / f"{stem}.png"
            out_abs = paths.interim / out_rel
            out_abs.parent.mkdir(parents=True, exist_ok=True)
            out_abs.write_bytes(encode_png(img))

            meta = {
                "orig_file": file_name,
                "health": str(row.get("health", "")),
                "subspecies": str(row.get("subspecies", "")),
                "caste": str(row.get("caste", "")),
                "pollen_carrying": bool(row.get("pollen_carrying", False)),
                "zip_code": str(row.get("zip code", "")),
                "location": str(row.get("location", "")),
                "date": str(row.get("date", "")),
                "time": str(row.get("time", "")),
            }
            records.append(
                MiteRecord(
                    id=rec_id,
                    source=MiteSource.BEEIMAGE,
                    image_path=out_rel,
                    label=MiteLabel(label_str),
                    split=Split(split_str),
                    meta=meta,
                )
            )
        except Exception as e:  # noqa: BLE001
            failures += 1
            LOG.warning("skip %s: %s", file_name, e)

    write_parquet(records, out_parquet)
    if limit is None:
        sentinel.write_text(current_hash)

    LOG.info(
        "wrote %d records → %s (failures=%d)", len(records), out_parquet, failures
    )
    return 0


# ---------- CLI -------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest BeeImage (Kaggle) into interim/.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    paths = load_paths(args.config)
    setup_logging(paths, args.dry_run)
    return ingest(paths, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
