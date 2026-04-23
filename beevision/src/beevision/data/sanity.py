"""Post-ingest sanity outputs.

Produces under ``interim/_sanity/``:

- ``stats.json``                          structured totals + per-source / per-split
                                          / per-class counts, plus image/mask size
                                          ranges (spot-checked per source).
- ``class_balance.png``                   stacked-bar of per-split class counts for
                                          every source that has a row-level label
                                          (deepbee_cls, beeimage, varroa).
- ``sample_grid_<source>_<split>.png``    8×8 random thumbnails drawn with a
                                          fixed per-source seed so reruns look the
                                          same. For ``deepbee_seg`` the mask is
                                          drawn as a transparent red overlay on top
                                          of the frame.

Prints a compact text summary to stdout for CI / quick inspection. Never writes
model output, never touches raw/; everything lives in interim/_sanity/.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless; sanity runs on login/batch nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from beevision.data.paths import Paths, load_paths
from beevision.data.splits import SOURCES

LOG = logging.getLogger("sanity")

GRID = 8                   # sample grid is GRID × GRID
THUMB = 128                # thumbnail side in px
SEED = 1337                # per-source + per-split jitter added


# ---------- Stats collection -----------------------------------------------


@dataclass
class SourceStats:
    source: str
    n_total: int = 0
    n_per_split: dict[str, int] = field(default_factory=dict)
    class_col: str | None = None
    per_split_class: dict[str, dict[str, int]] = field(default_factory=dict)
    image_sizes: dict[str, list[int]] = field(default_factory=dict)  # "w", "h"
    mask_class_values: list[int] = field(default_factory=list)


def _probe_image_sizes(df: pd.DataFrame, interim: Path, n: int = 16) -> dict[str, list[int]]:
    """Open up to ``n`` random images and record (w, h). Cheap, not exhaustive."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    widths: list[int] = []
    heights: list[int] = []
    for i in idx:
        p = interim / df.iloc[int(i)]["image_path"]
        if not p.exists():
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception as e:  # noqa: BLE001
            LOG.warning("size probe failed for %s: %s", p, e)
    return {"w": widths, "h": heights}


def collect_source_stats(paths: Paths, source: str) -> SourceStats | None:
    pq = paths.parquet_path(source)
    if not pq.exists():
        LOG.warning("skip %s: no parquet at %s", source, pq)
        return None
    df = pd.read_parquet(pq, engine="pyarrow")
    st = SourceStats(source=source, n_total=int(len(df)))
    st.n_per_split = {s: int(n) for s, n in df["split"].value_counts().to_dict().items()}
    if "label" in df.columns:
        st.class_col = "label"
        counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
        st.per_split_class = {
            str(s): {str(c): int(v) for c, v in row.items()} for s, row in counts.iterrows()
        }
    st.image_sizes = _probe_image_sizes(df, paths.interim)

    if source == "deepbee_seg" and "mask_path" in df.columns and not df.empty:
        # Decode first mask to sanity-check class values.
        p = paths.interim / df.iloc[0]["mask_path"]
        if p.exists():
            with Image.open(p) as m:
                arr = np.asarray(m)
            st.mask_class_values = sorted(int(v) for v in np.unique(arr))
    return st


# ---------- Plot helpers ---------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(stats: list[SourceStats], out_path: Path) -> Path | None:
    """Stacked bar: (train/val/test) × class per source."""
    classed = [s for s in stats if s.class_col]
    if not classed:
        return None
    n = len(classed)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, st in zip(axes[0], classed):
        splits_order = [x for x in ("train", "val", "test") if x in st.per_split_class]
        classes = sorted({c for row in st.per_split_class.values() for c in row})
        bottoms = np.zeros(len(splits_order), dtype=float)
        for cls in classes:
            heights = np.array([st.per_split_class[s].get(cls, 0) for s in splits_order], dtype=float)
            ax.bar(splits_order, heights, bottom=bottoms, label=cls)
            bottoms += heights
        ax.set_title(f"{st.source}  (n={st.n_total})")
        ax.set_ylabel("count")
        ax.legend(fontsize=7, loc="upper right", ncol=1)
    fig.suptitle("Per-split class balance", y=1.02, fontsize=12)
    _save(fig, out_path)
    return out_path


def _load_thumb(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((THUMB, THUMB))
    # Pad to THUMB×THUMB so the grid is tidy.
    canvas = Image.new("RGB", (THUMB, THUMB), color=(0, 0, 0))
    offset = ((THUMB - img.width) // 2, (THUMB - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


def _overlay_mask(img: Image.Image, mask_path: Path) -> Image.Image:
    with Image.open(mask_path) as m:
        if m.mode != "L":
            m = m.convert("L")
        m = m.resize(img.size, Image.Resampling.NEAREST)
        mask_arr = np.asarray(m)
    over = Image.new("RGBA", img.size, (255, 0, 0, 0))
    alpha = (mask_arr > 0).astype(np.uint8) * 110  # ~43% red where fg
    over_arr = np.dstack([
        np.full(alpha.shape, 255, dtype=np.uint8),
        np.zeros(alpha.shape, dtype=np.uint8),
        np.zeros(alpha.shape, dtype=np.uint8),
        alpha,
    ])
    over = Image.fromarray(over_arr, mode="RGBA")
    base = img.convert("RGBA")
    blended = Image.alpha_composite(base, over)
    return blended.convert("RGB")


def plot_sample_grid(
    df: pd.DataFrame,
    interim: Path,
    source: str,
    split: str,
    out_path: Path,
) -> Path | None:
    if df.empty:
        return None
    rng = random.Random(f"{SEED}:{source}:{split}")
    n = min(GRID * GRID, len(df))
    picks = rng.sample(range(len(df)), n)
    tiles: list[Image.Image] = []
    for i in picks:
        row = df.iloc[i]
        img_abs = interim / row["image_path"]
        if not img_abs.exists():
            continue
        try:
            thumb = _load_thumb(img_abs)
            if source == "deepbee_seg" and "mask_path" in row and row["mask_path"]:
                mp = interim / row["mask_path"]
                if mp.exists():
                    thumb = _overlay_mask(thumb, mp)
            tiles.append(thumb)
        except Exception as e:  # noqa: BLE001
            LOG.warning("thumb failed %s: %s", img_abs, e)

    if not tiles:
        return None
    # Fill missing tiles with black to keep grid rectangular.
    while len(tiles) < GRID * GRID:
        tiles.append(Image.new("RGB", (THUMB, THUMB), color=(0, 0, 0)))
    grid_img = Image.new("RGB", (THUMB * GRID, THUMB * GRID), color=(0, 0, 0))
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, GRID)
        grid_img.paste(tile, (c * THUMB, r * THUMB))

    # Add a small header with source/split and count.
    fig, ax = plt.subplots(figsize=(GRID, GRID), dpi=120)
    ax.imshow(np.asarray(grid_img))
    ax.set_title(f"{source} / {split}  (n={len(df)})")
    ax.axis("off")
    _save(fig, out_path)
    return out_path


# ---------- Orchestration --------------------------------------------------


def build_summary(stats: list[SourceStats]) -> dict[str, Any]:
    by_src: dict[str, Any] = {}
    for st in stats:
        size_info: dict[str, Any] = {}
        for k, vals in st.image_sizes.items():
            if vals:
                size_info[k] = {
                    "min": int(min(vals)),
                    "max": int(max(vals)),
                    "median": int(np.median(vals)),
                    "n_probed": len(vals),
                }
        entry: dict[str, Any] = {
            "n_total": st.n_total,
            "n_per_split": st.n_per_split,
            "image_size_probe": size_info,
        }
        if st.class_col:
            entry["class_col"] = st.class_col
            entry["per_split_class"] = st.per_split_class
        if st.mask_class_values:
            entry["mask_class_values"] = st.mask_class_values
        by_src[st.source] = entry
    by_src["__totals__"] = {
        "n_records": sum(st.n_total for st in stats),
        "sources_present": sorted(st.source for st in stats),
    }
    return by_src


def _print_summary(stats: list[SourceStats]) -> None:
    print("=" * 72)
    print(f"BeeVision sanity — {len(stats)} source(s)")
    print("=" * 72)
    for st in stats:
        print(f"[{st.source}] n={st.n_total}  splits={st.n_per_split}")
        if st.class_col:
            for sp, counts in st.per_split_class.items():
                pieces = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                print(f"    {sp}: {pieces}")
        if st.image_sizes.get("w"):
            w = st.image_sizes["w"]
            h = st.image_sizes["h"]
            print(f"    img size (n={len(w)} probed): w∈[{min(w)},{max(w)}] h∈[{min(h)},{max(h)}]")
        if st.mask_class_values:
            print(f"    mask class values: {st.mask_class_values}")
    print("=" * 72)


def run(paths: Paths, dry_run: bool = False) -> int:
    paths.ensure()
    sanity_dir = paths.sanity
    sanity_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("collecting stats…")
    stats = [s for s in (collect_source_stats(paths, src) for src in SOURCES) if s is not None]
    if not stats:
        LOG.error("no sources had a parquet on disk; nothing to do")
        return 2

    _print_summary(stats)

    if dry_run:
        LOG.info("dry-run: skipping stats.json / PNG writes")
        return 0

    # stats.json
    summary = build_summary(stats)
    (sanity_dir / "stats.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    LOG.info("wrote %s", sanity_dir / "stats.json")

    # class_balance.png
    p = plot_class_balance(stats, sanity_dir / "class_balance.png")
    if p:
        LOG.info("wrote %s", p)

    # sample_grid per (source, split)
    for st in stats:
        df = pd.read_parquet(paths.parquet_path(st.source), engine="pyarrow")
        for split in ("train", "val", "test"):
            sub = df[df["split"] == split]
            if sub.empty:
                continue
            p = plot_sample_grid(
                sub, paths.interim, st.source, split,
                sanity_dir / f"sample_grid_{st.source}_{split}.png",
            )
            if p:
                LOG.info("wrote %s", p)

    return 0


# ---------- CLI ------------------------------------------------------------


def _setup_logging(paths: Paths) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "sanity.log"),
        ],
        force=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate sanity stats + plots.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = load_paths(args.config)
    _setup_logging(paths)
    return run(paths, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
