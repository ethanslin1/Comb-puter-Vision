"""Ingest DS-COMB segmentation (Mendeley) → interim/deepbee_seg.parquet.

What's on disk
--------------
``raw/deepbee-segmentation/`` ships ``Images/`` and ``Annotations/`` with 61
matched filenames (full-resolution JPGs, 6000×4000). The annotations are
3-channel JPEGs but are effectively grayscale (R≈G≈B per-pixel) with a
binary palette: ~50% pixels at gray≈0 (background) and ~50% at gray≈255
(comb cells). Only ~0.2% of pixels sit in the ambiguous mid-gray band and
are produced by JPEG boundary artifacts, not real intermediate classes.

This is consistent with the DeepBee paper: segmentation is binary
foreground detection; per-cell classification is a separate task fed by
``ingest_deepbee_cls``. We therefore emit *binary* masks here with
``classes = [background, comb]`` and fail loudly if a frame's ambiguous
pixel fraction exceeds the threshold in configs/data.yaml.

What this module produces
-------------------------
For every frame we emit a pair::

    interim/images/deepbee_seg/<split>/<stem>.png   # resized RGB, short-edge frame_short_edge
    interim/masks/deepbee_seg/<split>/<stem>.png    # single-channel uint8 class indices {0, 1}
    interim/deepbee_seg.parquet                     # FrameRecord rows

Splits: stratified 70/15/15 by foreground-fraction tercile (seed=1337).

Spec non-negotiables honored here: deterministic output bytes, idempotent
skip via a sha256 sentinel, no absolute paths (everything via Paths), a
``--dry-run`` flag, and python logging to stdout + ``interim/_logs/``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from beevision.data.paths import Paths, load_paths
from beevision.data.schema import (
    FrameRecord,
    FrameSource,
    Split,
    write_parquet,
)

SOURCE = "deepbee_seg"
LOG = logging.getLogger(f"ingest_{SOURCE}")


# ---------- Logging ---------------------------------------------------------


def setup_logging(paths: Paths, dry_run: bool) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if not dry_run:
        handlers.append(logging.FileHandler(paths.logs / f"ingest_{SOURCE}.log"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


# ---------- Mask decoding ---------------------------------------------------


def decode_mask(
    arr: np.ndarray,
    threshold_gray: int,
    channel_consistency_tol: int,
) -> tuple[np.ndarray, dict]:
    """Convert a (H, W, 3) uint8 annotation array to a (H, W) uint8 class mask.

    Returns ``(mask, stats)``. ``mask`` has values in ``{0, 1}`` (background,
    comb). ``stats`` carries per-frame QA numbers: fraction of ambiguous
    mid-gray pixels and channel-consistency extremes. The caller is expected
    to abort if ``stats["ambiguous_frac"]`` exceeds the configured ceiling.

    Raises ``ValueError`` if R/G/B channels disagree by more than
    ``channel_consistency_tol`` anywhere — the dataset's masks are supposed
    to be grayscale-stored-as-RGB, so any true color content is a red flag.
    """
    if arr.ndim == 2:
        gray = arr.astype(np.int16)
        p99_channel_diff = 0
    else:
        if arr.shape[-1] < 3:
            raise ValueError(f"expected ≥3 channels, got shape {arr.shape}")
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        # JPEG chroma subsampling leaves a handful of pixels with R≠B at
        # sharp boundaries even when the source was pure grayscale. Check
        # the 99th-percentile diff rather than the max, so a few artifact
        # pixels don't fail a whole dataset, but any real color content
        # (which would affect >1% of pixels) still aborts.
        diffs = np.maximum(np.maximum(np.abs(r - g), np.abs(r - b)), np.abs(g - b))
        p99_channel_diff = int(np.percentile(diffs, 99))
        if p99_channel_diff > channel_consistency_tol:
            raise ValueError(
                f"annotation channels disagree at p99 by {p99_channel_diff} "
                f"(tol={channel_consistency_tol}). Expected grayscale-in-RGB; "
                "true color content suggests an unexpected palette."
            )
        gray = ((r + g + b) // 3).astype(np.int16)

    # Ambiguous band: [32, 223] (everything not clearly bg or clearly fg).
    total = gray.size
    ambiguous = int(((gray >= 32) & (gray <= 223)).sum())
    mask = (gray >= int(threshold_gray)).astype(np.uint8)
    stats = {
        "ambiguous_frac": ambiguous / total,
        "p99_channel_diff": p99_channel_diff,
        "fg_frac": float(mask.mean()),
    }
    return mask, stats


# ---------- Image/mask resize ----------------------------------------------


def resize_short_edge(img: Image.Image, short: int, *, is_mask: bool) -> Image.Image:
    """Resize so the short edge equals ``short``, preserving aspect ratio.

    Uses Lanczos for images and nearest-neighbour for masks (we must not
    interpolate class indices).
    """
    w, h = img.size
    s = min(w, h)
    if s == short:
        return img
    scale = short / float(s)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    return img.resize(new_size, resample)


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


# ---------- Splits ---------------------------------------------------------


def stratify_by_fg_tercile(
    frames: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign train/val/test, stratified on foreground-fraction tercile.

    ``frames`` must carry a numeric ``fg_frac`` column. Output is the input
    frame with a new ``split`` column.
    """
    if abs(1 - val_fraction - test_fraction) <= 0:
        raise ValueError("val_fraction + test_fraction must be < 1")
    tercile = pd.qcut(frames["fg_frac"], q=3, labels=["lo", "mid", "hi"], duplicates="drop")
    y = tercile.astype(str).to_numpy()
    idx = frames.index.to_numpy()

    # First carve test, then val from remainder.
    idx_trainval, idx_test, y_trainval, _y_test = train_test_split(
        idx, y, test_size=test_fraction, stratify=y, random_state=seed
    )
    rel_val = val_fraction / (1 - test_fraction)
    idx_train, idx_val, *_ = train_test_split(
        idx_trainval, y_trainval, test_size=rel_val, stratify=y_trainval, random_state=seed
    )

    splits = pd.Series(index=frames.index, dtype=object)
    splits.loc[idx_train] = "train"
    splits.loc[idx_val] = "val"
    splits.loc[idx_test] = "test"
    out = frames.copy()
    out["split"] = splits.to_numpy()
    return out


# ---------- Hashing --------------------------------------------------------


def hash_config(
    image_names: list[str],
    frame_short_edge: int,
    classes: list[str],
    threshold_gray: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> str:
    h = hashlib.sha256()
    h.update("|".join(sorted(image_names)).encode())
    h.update(
        f"|edge={frame_short_edge}|classes={classes}|thr={threshold_gray}"
        f"|val={val_fraction}|test={test_fraction}|seed={seed}".encode()
    )
    return h.hexdigest()


def sentinel_path(paths: Paths) -> Path:
    paths.cache.mkdir(parents=True, exist_ok=True)
    return paths.cache / f"{SOURCE}.ingest.hash"


# ---------- Core ingest ----------------------------------------------------


def _pair_filenames(images_dir: Path, ann_dir: Path) -> list[tuple[Path, Path]]:
    imgs = {p.name: p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    anns = {p.name: p for p in ann_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    common = sorted(set(imgs) & set(anns))
    missing_ann = sorted(set(imgs) - set(anns))
    missing_img = sorted(set(anns) - set(imgs))
    if missing_ann:
        LOG.warning("images with no matching annotation (%d): %s", len(missing_ann), missing_ann[:5])
    if missing_img:
        LOG.warning("annotations with no matching image (%d): %s", len(missing_img), missing_img[:5])
    return [(imgs[n], anns[n]) for n in common]


def ingest(paths: Paths, dry_run: bool = False, limit: int | None = None) -> int:
    paths.ensure()
    raw_dir = paths.raw_source_dir(SOURCE)
    src_cfg = paths.sources[SOURCE]

    images_dir = raw_dir / src_cfg["images_dir"]
    ann_dir = raw_dir / src_cfg["annotations_dir"]
    for p in (images_dir, ann_dir):
        if not p.exists():
            LOG.error("required directory missing: %s", p)
            return 2

    classes: list[str] = list(src_cfg["classes"])
    threshold_gray: int = int(src_cfg["threshold_gray"])
    abort_frac_above: float = float(src_cfg["abort_frac_above"])
    channel_tol: int = int(src_cfg["channel_consistency_tol"])
    val_fraction: float = float(src_cfg["val_fraction"])
    test_fraction: float = float(src_cfg["test_fraction"])
    frame_short_edge: int = int(paths.cfg["image_sizes"]["frame_short_edge"])
    seed = int(paths.cfg["seed"])

    pairs = _pair_filenames(images_dir, ann_dir)
    LOG.info("matched %d image/annotation pairs", len(pairs))
    if not pairs:
        LOG.error("no image/annotation pairs found")
        return 2

    current_hash = hash_config(
        [p[0].name for p in pairs],
        frame_short_edge,
        classes,
        threshold_gray,
        val_fraction,
        test_fraction,
        seed,
    )
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

    # Pass 1: decode masks once to compute fg_frac for stratification. We
    # don't write yet in order to validate all frames up front (spec's
    # "fail loudly" rule) before emitting anything.
    LOG.info("pass 1: validating + measuring foreground fraction")
    rows = []
    for img_path, ann_path in tqdm(pairs, desc=f"{SOURCE}/qa", unit="frm"):
        ann = Image.open(ann_path)
        ann.load()
        if ann.mode != "RGB":
            ann = ann.convert("RGB")
        arr = np.asarray(ann)
        try:
            _, stats = decode_mask(arr, threshold_gray, channel_tol)
        except ValueError as e:
            LOG.error("FAIL LOUD: %s: %s", ann_path.name, e)
            return 3
        if stats["ambiguous_frac"] > abort_frac_above:
            LOG.error(
                "FAIL LOUD: %s has ambiguous_frac=%.4f > %.4f",
                ann_path.name, stats["ambiguous_frac"], abort_frac_above,
            )
            return 3
        rows.append(
            {
                "image_path": img_path,
                "ann_path": ann_path,
                "image_name": img_path.name,
                "fg_frac": stats["fg_frac"],
                "ambiguous_frac": stats["ambiguous_frac"],
                "p99_channel_diff": stats["p99_channel_diff"],
            }
        )

    frames = pd.DataFrame(rows).sort_values("image_name", kind="mergesort").reset_index(drop=True)
    LOG.info(
        "fg_frac range [%.3f, %.3f]; ambiguous_frac max=%.4f",
        frames["fg_frac"].min(),
        frames["fg_frac"].max(),
        frames["ambiguous_frac"].max(),
    )

    frames = stratify_by_fg_tercile(
        frames, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )
    counts = frames.groupby("split").size()
    LOG.info("split counts:\n%s", counts.to_string())

    if limit is not None:
        frames = frames.head(limit).reset_index(drop=True)
    LOG.info("processing %d frames (limit=%s)", len(frames), limit)

    if dry_run:
        LOG.info("dry-run: would write %d frame+mask pairs → %s", len(frames), out_parquet)
        for _, row in frames.head(5).iterrows():
            LOG.info(
                "DRY %s fg=%.3f ambig=%.4f split=%s",
                row["image_name"], row["fg_frac"], row["ambiguous_frac"], row["split"],
            )
        return 0

    img_root = paths.interim_images_dir(SOURCE)
    mask_root = paths.interim_masks_dir(SOURCE)
    img_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []
    failures = 0
    for _, row in tqdm(list(frames.iterrows()), desc=f"{SOURCE}/write", unit="frm"):
        try:
            img_path: Path = row["image_path"]
            ann_path: Path = row["ann_path"]
            split_str: str = row["split"]
            stem = img_path.stem

            img = Image.open(img_path)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            ann = Image.open(ann_path)
            ann.load()
            if ann.mode != "RGB":
                ann = ann.convert("RGB")

            arr = np.asarray(ann)
            mask_arr, stats = decode_mask(arr, threshold_gray, channel_tol)
            mask_img = Image.fromarray(mask_arr, mode="L")

            img_resized = resize_short_edge(img, frame_short_edge, is_mask=False)
            mask_resized = resize_short_edge(mask_img, frame_short_edge, is_mask=True)

            img_rel = Path("images") / SOURCE / split_str / f"{stem}.png"
            mask_rel = Path("masks") / SOURCE / split_str / f"{stem}.png"
            img_abs = paths.interim / img_rel
            mask_abs = paths.interim / mask_rel
            img_abs.parent.mkdir(parents=True, exist_ok=True)
            mask_abs.parent.mkdir(parents=True, exist_ok=True)
            img_abs.write_bytes(encode_png(img_resized))
            mask_abs.write_bytes(encode_png(mask_resized))

            meta = {
                "orig_image": img_path.name,
                "orig_size_wh": [img.size[0], img.size[1]],
                "processed_size_wh": [img_resized.size[0], img_resized.size[1]],
                "fg_frac": stats["fg_frac"],
                "ambiguous_frac_original": stats["ambiguous_frac"],
                "p99_channel_diff": stats["p99_channel_diff"],
                "classes": classes,
            }
            records.append(
                FrameRecord(
                    id=f"{SOURCE}:{stem}",
                    source=FrameSource.DEEPBEE_SEG,
                    image_path=img_rel,
                    mask_path=mask_rel,
                    split=Split(split_str),
                    meta=meta,
                )
            )
        except Exception as e:  # noqa: BLE001
            failures += 1
            LOG.warning("skip %s: %s", row.get("image_name"), e)

    write_parquet(records, out_parquet)
    if limit is None:
        sentinel.write_text(current_hash)
    LOG.info("wrote %d records → %s (failures=%d)", len(records), out_parquet, failures)
    return 0


# ---------- CLI ------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest DS-COMB segmentation into interim/.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    paths = load_paths(args.config)
    setup_logging(paths, args.dry_run)
    return ingest(paths, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
