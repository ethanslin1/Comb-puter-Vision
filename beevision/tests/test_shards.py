"""Unit tests for beevision.data.shard_writer."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from beevision.data.shard_writer import (
    Sample,
    _clamp_tarinfo,
    _row_to_sample,
    write_stream,
)


# ---------- Deterministic tar headers --------------------------------------


def test_clamp_tarinfo_zeroes_metadata() -> None:
    info = tarfile.TarInfo(name="x.png")
    info.mtime = 123456789
    info.uid = 1000
    info.gid = 1000
    info.uname = "me"
    info.gname = "mygrp"
    _clamp_tarinfo(info)
    assert info.mtime == 0
    assert info.uid == 0
    assert info.gid == 0
    assert info.uname == ""
    assert info.gname == ""


# ---------- Sample assembly ------------------------------------------------


def _make_interim(tmp: Path) -> Path:
    """Write one tiny PNG + one tiny mask under an 'interim' layout."""
    interim = tmp / "interim"
    img_dir = interim / "images/beeimage/train"
    mask_dir = interim / "masks/deepbee_seg/train"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    img.save(img_dir / "a.png")
    img.save(img_dir / "b.png")
    m = Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8), mode="L")
    m.save(mask_dir / "a.png")
    return interim


def test_row_to_sample_classification_emits_label_txt(tmp_path: Path) -> None:
    interim = _make_interim(tmp_path)
    row = {
        "id": "beeimage:a",
        "source": "beeimage",
        "split": "train",
        "image_path": "images/beeimage/train/a.png",
        "label": "mite",
        "meta": {"orig_file": "a.png"},
    }
    s = _row_to_sample(row, "beeimage", interim)
    assert s.target_name == "label.txt"
    assert s.target_bytes == b"mite"
    # Meta JSON is stable + sorted.
    meta = json.loads(s.meta_bytes)
    assert meta["id"] == "beeimage:a"
    assert meta["label"] == "mite"


def test_row_to_sample_segmentation_emits_mask_png(tmp_path: Path) -> None:
    interim = _make_interim(tmp_path)
    row = {
        "id": "deepbee_seg:a",
        "source": "deepbee_seg",
        "split": "train",
        "image_path": "images/beeimage/train/a.png",   # reuse the png
        "mask_path": "masks/deepbee_seg/train/a.png",
        "meta": {"classes": ["background", "comb"]},
    }
    s = _row_to_sample(row, "deepbee_seg", interim)
    assert s.target_name == "mask.png"
    # The bytes must be the exact PNG file on disk.
    mask_abs = interim / row["mask_path"]
    assert s.target_bytes == mask_abs.read_bytes()


# ---------- Stream writing -------------------------------------------------


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_samples(n: int) -> list[Sample]:
    samples = []
    for i in range(n):
        samples.append(
            Sample(
                key=f"k{i:03d}",
                image_bytes=b"\x89PNG\r\n\x1a\n" + bytes([i % 256]) * 50,
                target_name="label.txt",
                target_bytes=f"lbl{i % 3}".encode(),
                meta_bytes=json.dumps({"i": i}, sort_keys=True).encode(),
            )
        )
    return samples


def test_write_stream_rolls_shards_on_size_cap(tmp_path: Path) -> None:
    samples = _make_samples(20)
    shards = write_stream(
        iter(samples),
        out_dir=tmp_path,
        source="toy",
        split="train",
        shard_max_bytes=4096,
    )
    assert len(shards) >= 2, shards
    # Each shard has its sidecar manifest.
    for s in shards:
        assert s.exists()
        assert s.with_suffix(".tar.sha256").exists()
    # Union of __key__ names across shards == input.
    names = []
    for s in shards:
        with tarfile.open(s) as tf:
            for m in tf.getmembers():
                names.append(m.name)
    for i in range(20):
        assert f"k{i:03d}.image.png" in names


def test_write_stream_produces_byte_identical_tars_on_rerun(tmp_path: Path) -> None:
    samples_a = _make_samples(6)
    out_a = tmp_path / "a"
    shards_a = write_stream(iter(samples_a), out_dir=out_a, source="toy",
                            split="train", shard_max_bytes=1 << 20)

    samples_b = _make_samples(6)
    out_b = tmp_path / "b"
    shards_b = write_stream(iter(samples_b), out_dir=out_b, source="toy",
                            split="train", shard_max_bytes=1 << 20)

    assert [p.name for p in shards_a] == [p.name for p in shards_b]
    for pa, pb in zip(shards_a, shards_b):
        assert _sha256(pa) == _sha256(pb), f"{pa.name} differs between runs"


def test_write_stream_is_atomic_no_partial_left_behind(tmp_path: Path) -> None:
    samples = _make_samples(4)
    shards = write_stream(iter(samples), out_dir=tmp_path, source="toy",
                          split="train", shard_max_bytes=1 << 20)
    assert shards, "expected at least one shard"
    # After successful write, no .partial files remain.
    assert not any(tmp_path.glob("*.partial")), list(tmp_path.iterdir())
