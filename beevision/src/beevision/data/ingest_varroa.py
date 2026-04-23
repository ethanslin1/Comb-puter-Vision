"""Ingest VarroaDataset (Zenodo 4085044) → interim/varroa.parquet + PNGs.

Input layout (under raw/varroa-mite-detection/)::

    gt.csv                          # space-delimited: <path> <count> [x1 y1 x2 y2]*
    train.zip val.zip test.zip      # per-split archives of bee crops

Output::

    interim/images/varroa/<split>/<id>.png      # normalized 224-short-edge PNG
    interim/varroa.parquet                      # unified MiteRecord rows

Rules honored
-------------
- Respect the zip-provided train/val/test split (the first path segment in gt.csv).
- Binary label: count == 0 → no_mite; count >= 1 → mite. Original boxes kept in meta.
- Decode-once: PIL opens each zip entry in memory, resizes short-edge to ``crop``,
  saves deterministic PNG. No temp extraction directory.
- Idempotent: a sha256 of gt.csv + crop size is stored in cache/varroa.ingest.hash;
  the run short-circuits if the hash matches and the parquet already exists.
- --dry-run prints a preview and exits without writing anything.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm

from beevision.data.paths import Paths, load_paths
from beevision.data.schema import (
    MiteLabel,
    MiteRecord,
    MiteSource,
    Split,
    write_parquet,
)

SOURCE = "varroa"
LOG = logging.getLogger(f"ingest_{SOURCE}")

BBox = Tuple[int, int, int, int]
GtRow = Tuple[str, int, List[BBox]]


# ---------- Logging ---------------------------------------------------------


def setup_logging(paths: Paths, dry_run: bool) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if not dry_run:
        handlers.append(logging.FileHandler(paths.logs / f"ingest_{SOURCE}.log"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


# ---------- gt.csv parsing --------------------------------------------------


def parse_gt_line(line: str) -> GtRow:
    """Parse one gt.csv row. Format::

        <path> <count> [x1 y1 x2 y2]*
    """
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"malformed gt row: {line!r}")
    path = parts[0]
    count = int(parts[1])
    rest = list(map(int, parts[2:]))
    if len(rest) % 4 != 0:
        raise ValueError(f"bbox coords not divisible by 4 in: {line!r}")
    boxes: list[BBox] = [(rest[i], rest[i + 1], rest[i + 2], rest[i + 3]) for i in range(0, len(rest), 4)]
    if count != len(boxes) and count > 0:
        # Ground truth may annotate fewer boxes than counted bees; warn but trust the count.
        LOG.debug("count=%d but %d boxes in %s", count, len(boxes), path)
    return path, count, boxes


def read_gt_csv(gt_path: Path) -> List[GtRow]:
    rows: list[GtRow] = []
    with gt_path.open("r") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(parse_gt_line(line))
    rows.sort(key=lambda r: r[0])
    return rows


def infer_split(path: str) -> Split:
    head = path.split("/", 1)[0]
    return {"train": Split.TRAIN, "val": Split.VAL, "test": Split.TEST}[head]


# ---------- Image handling --------------------------------------------------


def resize_short_edge(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    if s == short:
        return img
    scale = short / float(s)
    new = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new, Image.Resampling.LANCZOS)


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


# ---------- Hashing ---------------------------------------------------------


def hash_config(gt_path: Path, target_size: int) -> str:
    h = hashlib.sha256()
    h.update(gt_path.read_bytes())
    h.update(f"|crop={target_size}".encode())
    return h.hexdigest()


def sentinel_path(paths: Paths) -> Path:
    paths.cache.mkdir(parents=True, exist_ok=True)
    return paths.cache / f"{SOURCE}.ingest.hash"


# ---------- Core ingest ------------------------------------------------------


def make_record_id(rel_path: str) -> str:
    """Stable id: replace slashes/extensions; no random parts."""
    stem = rel_path.rsplit(".", 1)[0]
    return f"{SOURCE}:" + stem.replace("/", "_")


def ingest(paths: Paths, dry_run: bool = False, limit: int | None = None) -> int:
    paths.ensure()
    raw_dir = paths.raw_source_dir(SOURCE)
    src_cfg = paths.sources[SOURCE]
    gt_path = raw_dir / src_cfg["labels_file"]
    if not gt_path.exists():
        LOG.error("gt.csv missing: %s", gt_path)
        return 2

    target_size = int(paths.cfg["image_sizes"]["crop"])
    current_hash = hash_config(gt_path, target_size)
    out_parquet = paths.parquet_path(SOURCE)
    sentinel = sentinel_path(paths)

    if (
        not dry_run
        and sentinel.exists()
        and sentinel.read_text().strip() == current_hash
        and out_parquet.exists()
    ):
        LOG.info("up to date (hash=%s…); skipping", current_hash[:12])
        return 0

    rows = read_gt_csv(gt_path)
    # gt.csv has a handful of exact-duplicate rows (same video+frame+bee_id+roi+
    # label). Keeping both would produce duplicate record ids and identical
    # output PNGs, so de-duplicate on the generated record id while preserving
    # first-seen order (read_gt_csv already sorts deterministically).
    seen_ids: set[str] = set()
    deduped: list = []
    for r in rows:
        rel_path = r[0]  # (rel_path, count, boxes)
        rid = make_record_id(rel_path)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        deduped.append(r)
    n_drop = len(rows) - len(deduped)
    if n_drop:
        LOG.info("dropped %d duplicate gt row(s)", n_drop)
    rows = deduped
    if limit is not None:
        rows = rows[:limit]
    LOG.info("gt rows: %d (limit=%s)", len(rows), limit)

    counts_by_split: dict[str, dict[str, int]] = {
        s: {"mite": 0, "no_mite": 0} for s in ("train", "val", "test")
    }

    if dry_run:
        for r in rows[:5]:
            LOG.info("DRY %s count=%d boxes=%d split=%s", r[0], r[1], len(r[2]), infer_split(r[0]).value)
        LOG.info("dry-run: would process %d rows → %s", len(rows), out_parquet)
        return 0

    # Open all three zips once.
    zips: dict[str, zipfile.ZipFile] = {}
    for split in ("train", "val", "test"):
        zp = raw_dir / f"{split}.zip"
        if not zp.exists():
            LOG.error("missing zip: %s", zp)
            return 2
        zips[split] = zipfile.ZipFile(zp)

    out_root = paths.interim_images_dir(SOURCE)
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[MiteRecord] = []
    failures = 0
    try:
        for rel_path, count, boxes in tqdm(rows, desc=SOURCE, unit="img"):
            try:
                split = infer_split(rel_path)
                with zips[split.value].open(rel_path) as fh:
                    img = Image.open(io.BytesIO(fh.read()))
                    img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = resize_short_edge(img, target_size)

                rec_id = make_record_id(rel_path)
                out_rel = Path("images") / SOURCE / split.value / f"{rec_id.split(':', 1)[1]}.png"
                out_abs = paths.interim / out_rel
                out_abs.parent.mkdir(parents=True, exist_ok=True)
                out_abs.write_bytes(encode_png(img))

                label = MiteLabel.MITE if count > 0 else MiteLabel.NO_MITE
                counts_by_split[split.value][label.value] += 1

                records.append(
                    MiteRecord(
                        id=rec_id,
                        source=MiteSource.VARROA,
                        image_path=out_rel,
                        label=label,
                        split=split,
                        meta={
                            "orig_path": rel_path,
                            "mite_count": int(count),
                            "boxes": [list(b) for b in boxes],
                        },
                    )
                )
            except Exception as e:  # noqa: BLE001
                failures += 1
                LOG.warning("skip %s: %s", rel_path, e)
    finally:
        for z in zips.values():
            z.close()

    write_parquet(records, out_parquet)
    # Only stamp the sentinel when we processed the full dataset; otherwise a
    # --limit run would short-circuit the next full run.
    if limit is None:
        sentinel.write_text(current_hash)

    total = len(records)
    LOG.info("wrote %d records → %s (failures=%d)", total, out_parquet, failures)
    for split, d in counts_by_split.items():
        LOG.info("  %-5s mite=%d  no_mite=%d", split, d["mite"], d["no_mite"])
    return 0


# ---------- CLI -------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest VarroaDataset into interim/.")
    ap.add_argument("--config", required=True, help="Path to configs/data.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N rows (debug)")
    args = ap.parse_args()
    paths = load_paths(args.config)
    setup_logging(paths, args.dry_run)
    return ingest(paths, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
