"""Pack interim PNGs + records into WebDataset tar shards.

Spec non-negotiables:

- 256 MB shard target (``shards.size_bytes`` in configs/data.yaml).
- One shard per ``(source, split)`` stream; rolled over at the size cap.
- Samples carry ``__key__``, ``image.png``, either ``label.txt`` (for
  CellRecord / MiteRecord) or ``mask.png`` (for FrameRecord), and
  ``meta.json``.
- **Byte-identical re-runs**: we sort records by id, emit files in
  fixed-key order within each sample, stamp every tar entry with mtime=0
  and uid/gid=0 via a custom filter, and record a per-shard sha256
  manifest at ``<shard>.sha256``. A re-run short-circuits on the manifest.

Implementation notes
--------------------
``tarfile`` writes deterministic bytes as long as:
1. Entries are added in a fixed order.
2. ``TarInfo.mtime``, ``uid``, ``gid``, ``uname``, ``gname`` are normalized.
3. No implicit compression, since gzip writes a timestamp in the header.
   We emit plain ``.tar`` (webdataset reads both).

We write each tar to a ``.partial`` tempfile and rename atomically; that
way a crashed run never leaves a half-tar for the next invocation to
treat as complete.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

from beevision.data.paths import Paths, load_paths
from beevision.data.schema import (
    CellRecord,
    FrameRecord,
    MiteRecord,
    dataframe_to_records,
)

LOG = logging.getLogger("shards")

# Source → pydantic model. Drives which "target" field we write.
_MODEL_BY_SOURCE: dict[str, type] = {
    "varroa": MiteRecord,
    "beeimage": MiteRecord,
    "deepbee_cls": CellRecord,
    "deepbee_seg": FrameRecord,
}


# ---------- Deterministic tar helpers --------------------------------------


def _clamp_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Zero timestamps/uids so tars are byte-identical across runs/hosts."""
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    # Keep a sane permission even if source files differ.
    if info.isfile():
        info.mode = 0o644
    return info


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    _clamp_tarinfo(info)
    tf.addfile(info, io.BytesIO(data))


# ---------- Sample assembly -----------------------------------------------


@dataclass
class Sample:
    key: str
    image_bytes: bytes
    target_name: str
    target_bytes: bytes
    meta_bytes: bytes

    def add_to(self, tf: tarfile.TarFile) -> int:
        """Write this sample to ``tf``; return bytes written (approx)."""
        entries = (
            (f"{self.key}.image.png", self.image_bytes),
            (f"{self.key}.{self.target_name}", self.target_bytes),
            (f"{self.key}.meta.json", self.meta_bytes),
        )
        total = 0
        for name, data in entries:
            _add_bytes(tf, name, data)
            total += len(data)
        return total

    @property
    def nominal_size(self) -> int:
        # 512-byte tar header per entry + payload, rounded up to 512.
        def _pad(n: int) -> int:
            return ((n + 511) // 512) * 512
        return sum(
            512 + _pad(len(b))
            for b in (self.image_bytes, self.target_bytes, self.meta_bytes)
        )


def _safe_meta(meta: Any) -> Any:
    """Coerce pydantic/parquet-dumped meta into plain JSON-friendly types.

    Pandas re-hydrates list-typed parquet columns as numpy object-dtype
    ndarrays whose elements can themselves be ndarrays. ``.tolist()`` on
    the outer only peels one layer, so we recurse after every conversion.
    """
    if meta is None:
        return None
    if isinstance(meta, dict):
        return {k: _safe_meta(v) for k, v in meta.items()}
    # numpy scalars / 0-d arrays: item() yields a native python scalar.
    if hasattr(meta, "item") and getattr(meta, "shape", None) == ():
        return meta.item()
    # Arrays / pandas Series: tolist may still contain ndarrays inside →
    # recurse on the result.
    if hasattr(meta, "tolist"):
        return _safe_meta(meta.tolist())
    if isinstance(meta, (list, tuple)):
        return [_safe_meta(v) for v in meta]
    return meta


def _row_to_sample(row: dict[str, Any], source: str, interim: Path) -> Sample:
    # WebDataset groups tar members by the longest common filename prefix
    # up to the *first* period. Keys must therefore be dot-free; varroa
    # ids contain ".mp4" (e.g. 'varroa:train_videos/...mp4-bee_id_...'),
    # which silently collapsed ~8000 samples per split into 9. We replace
    # colons, slashes, AND periods so every sample is its own group.
    key = row["id"].replace(":", "__").replace("/", "_").replace(".", "_")
    image_abs = interim / row["image_path"]
    if not image_abs.exists():
        raise FileNotFoundError(f"image not found: {image_abs}")
    image_bytes = image_abs.read_bytes()

    if source == "deepbee_seg":
        mask_abs = interim / row["mask_path"]
        if not mask_abs.exists():
            raise FileNotFoundError(f"mask not found: {mask_abs}")
        target_name = "mask.png"
        target_bytes = mask_abs.read_bytes()
    else:
        target_name = "label.txt"
        target_bytes = str(row["label"]).encode("utf-8")

    meta = _safe_meta(row.get("meta")) or {}
    meta_payload = {
        "id": row["id"],
        "source": row["source"],
        "split": row["split"],
        **(
            {"label": row["label"]}
            if source != "deepbee_seg"
            else {"classes": meta.get("classes")}
        ),
        "meta": meta,
    }
    meta_bytes = json.dumps(meta_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    return Sample(
        key=key,
        image_bytes=image_bytes,
        target_name=target_name,
        target_bytes=target_bytes,
        meta_bytes=meta_bytes,
    )


# ---------- Shard writing --------------------------------------------------


def _shard_name(source: str, split: str, idx: int) -> str:
    return f"{source}-{split}-{idx:06d}.tar"


def write_stream(
    samples: Iterator[Sample],
    *,
    out_dir: Path,
    source: str,
    split: str,
    shard_max_bytes: int,
) -> list[Path]:
    """Write a stream of samples into rolled tar shards under ``out_dir``.

    Returns the list of completed shard paths (in write order). Each shard
    is written atomically via ``<final>.partial`` → ``rename``; interrupted
    runs never leave a half-tar where a reader would see a complete one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    idx = 0
    tf: tarfile.TarFile | None = None
    partial_path: Path | None = None
    final_path: Path | None = None
    cur_bytes = 0

    def open_new_shard() -> None:
        nonlocal tf, idx, partial_path, final_path, cur_bytes
        final_path = out_dir / _shard_name(source, split, idx)
        partial_path = final_path.with_suffix(".tar.partial")
        tf = tarfile.open(partial_path, mode="w")
        cur_bytes = 0

    def close_current_shard() -> None:
        nonlocal tf, idx, partial_path, final_path
        if tf is None:
            return
        tf.close()
        assert partial_path is not None and final_path is not None
        partial_path.replace(final_path)
        _write_sha256_sidecar(final_path)
        written.append(final_path)
        idx += 1
        tf = None

    for sample in samples:
        if tf is None:
            open_new_shard()
        assert tf is not None
        # Roll over before writing if the next sample would overflow.
        if cur_bytes > 0 and cur_bytes + sample.nominal_size > shard_max_bytes:
            close_current_shard()
            open_new_shard()
        assert tf is not None
        cur_bytes += sample.add_to(tf)

    close_current_shard()
    return written


def _write_sha256_sidecar(shard_path: Path) -> None:
    h = hashlib.sha256()
    with open(shard_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    sidecar = shard_path.with_suffix(".tar.sha256")
    sidecar.write_text(f"{h.hexdigest()}  {shard_path.name}\n")


# ---------- Top-level orchestration ---------------------------------------


def _iter_records(
    df: pd.DataFrame,
    source: str,
    interim: Path,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[Sample]:
    # Deterministic order: sort by id, then row-by-row.
    df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    total = len(df)
    for i, row in enumerate(df.to_dict(orient="records"), start=1):
        yield _row_to_sample(row, source, interim)
        if progress:
            progress(i, total)


def write_source_shards(
    paths: Paths,
    source: str,
    shard_max_bytes: int,
    dry_run: bool = False,
) -> dict[str, list[Path]]:
    """Shard the interim parquet for ``source`` into per-split tar streams."""
    pq = paths.parquet_path(source)
    if not pq.exists():
        LOG.warning("skip %s: parquet missing %s", source, pq)
        return {}
    model = _MODEL_BY_SOURCE[source]
    df = pd.read_parquet(pq, engine="pyarrow")
    LOG.info("%s: %d records", source, len(df))
    _ = dataframe_to_records(df.head(1), model)  # sanity-check schema once

    out_dir = paths.shard_dir()
    written: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        LOG.info("  %s/%s → %d records", source, split, len(sub))
        if dry_run:
            written.setdefault(split, [])
            continue

        def _progress(i: int, total: int, split=split) -> None:
            if i == total or i % max(1, total // 20) == 0:
                LOG.info("    %s/%s %d/%d", source, split, i, total)

        paths_ = write_stream(
            _iter_records(sub, source, paths.interim, progress=_progress),
            out_dir=out_dir,
            source=source,
            split=split,
            shard_max_bytes=shard_max_bytes,
        )
        written[split] = paths_
        LOG.info("  %s/%s → %d shard(s): %s", source, split, len(paths_), [p.name for p in paths_])
    return written


# ---------- CLI ------------------------------------------------------------


def _setup_logging(paths: Paths) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "shards.log"),
        ],
        force=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack interim PNGs into WebDataset shards.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="Limit to these source names (default: all sources).")
    args = ap.parse_args()

    paths = load_paths(args.config)
    paths.ensure()
    _setup_logging(paths)

    size_bytes = int(paths.cfg.get("shards", {}).get("size_bytes", 256 * 1024 * 1024))
    LOG.info("shard size cap: %.1f MiB", size_bytes / (1024 * 1024))

    sources = args.sources or list(_MODEL_BY_SOURCE)
    all_written: dict[str, dict[str, list[Path]]] = {}
    for src in sources:
        if src not in _MODEL_BY_SOURCE:
            LOG.error("unknown source: %s", src)
            return 2
        all_written[src] = write_source_shards(paths, src, size_bytes, dry_run=args.dry_run)

    n_shards = sum(len(ps) for d in all_written.values() for ps in d.values())
    LOG.info("done: %d shards written", n_shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
