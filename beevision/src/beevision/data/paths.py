"""Filesystem layout resolution.

Everything the pipeline reads or writes is derived from one ``Paths`` object
built from ``configs/data.yaml`` + the ``BEEVISION_ROOT`` environment variable.
No module is allowed to hard-code absolute paths.

Layout (default; every key overridable via yaml)::

    $BEEVISION_ROOT/
    ├── raw/
    │   ├── varroa-mite-detection/
    │   ├── beeimage/
    │   ├── deepbee-classification/
    │   └── deepbee-segmentation/
    ├── interim/
    │   ├── <source>.parquet          # unified records per source
    │   ├── images/<source>/...       # decoded & normalized PNGs
    │   ├── _logs/
    │   ├── _debug/whitebalance/
    │   └── _sanity/
    ├── processed/
    │   └── shards/<source>-<split>-NNNNNN.tar
    └── cache/
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import yaml

__all__ = ["Paths", "load_paths", "load_config"]


# ---------- Config loading ---------------------------------------------------


def _expand(value: str) -> str:
    """Expand ${VAR} / $VAR using the current environment (USER, HOME, ...)."""
    return Template(value).safe_substitute(os.environ)


def load_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML config. Shallow utility; no interpolation here (see Paths)."""
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {config_path} did not parse to a mapping")
    return cfg


# ---------- Paths object -----------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout. Immutable once constructed."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    cache: Path
    logs: Path
    debug: Path
    sanity: Path

    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)

    # ---- Source-specific accessors ----------------------------------------

    def raw_source_dir(self, source: str) -> Path:
        """Return the raw/ directory for ``source`` (e.g. 'varroa')."""
        if source not in self.sources:
            raise KeyError(f"unknown source {source!r}; known: {sorted(self.sources)}")
        sub = self.sources[source].get("raw_dir", source)
        return self.raw / sub

    def parquet_path(self, source: str) -> Path:
        """Interim Parquet for unified records from ``source``."""
        return self.interim / f"{source}.parquet"

    def interim_images_dir(self, source: str) -> Path:
        """Normalized PNG images output directory for ``source``."""
        return self.interim / "images" / source

    def interim_masks_dir(self, source: str) -> Path:
        """Normalized class-index mask PNG directory for ``source``."""
        return self.interim / "masks" / source

    def shard_dir(self) -> Path:
        return self.processed / "shards"

    # ---- Ensure directories exist -----------------------------------------

    def ensure(self) -> "Paths":
        for p in (
            self.raw,
            self.interim,
            self.processed,
            self.cache,
            self.logs,
            self.debug,
            self.sanity,
            self.shard_dir(),
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


# ---------- Factory ----------------------------------------------------------


def load_paths(config_path: str | os.PathLike[str]) -> Paths:
    """Build a :class:`Paths` from a YAML config, honoring BEEVISION_ROOT."""
    cfg = load_config(config_path)

    root_cfg = cfg.get("root", {})
    env_var = root_cfg.get("env_var", "BEEVISION_ROOT")
    default = root_cfg.get("default", f"/oscar/scratch/{os.environ.get('USER', 'USER')}/beevision")
    root_str = os.environ.get(env_var, _expand(default))
    root = Path(root_str).expanduser().resolve()

    subs = cfg.get("subdirs", {})

    def _sub(name: str, default: str) -> Path:
        return root / subs.get(name, default)

    return Paths(
        root=root,
        raw=_sub("raw", "raw"),
        interim=_sub("interim", "interim"),
        processed=_sub("processed", "processed"),
        cache=_sub("cache", "cache"),
        logs=_sub("logs", "interim/_logs"),
        debug=_sub("debug", "interim/_debug"),
        sanity=_sub("sanity", "interim/_sanity"),
        sources=dict(cfg.get("sources", {})),
        cfg=cfg,
    )
