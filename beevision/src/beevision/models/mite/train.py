"""Mite-classifier training entry point.

Run::

    python -m beevision.models.mite.train --config configs/train_mite.yaml

The YAML config points at ``configs/data.yaml`` for paths and to a list of
sources (default: ``[varroa, beeimage]``). Per-source ``MiteClassificationDataset``
instances are concatenated; the train sampler is a ``WeightedRandomSampler``
built from per-row class weights summed across the concatenated set.

Checkpoints go under ``$BEEVISION_ROOT/processed/checkpoints/mite/``;
per-epoch metrics are appended as JSON lines to ``metrics.jsonl``.
``best.pt`` holds the best-by-val-balanced-accuracy checkpoint;
``last.pt`` holds the most recent.

Design notes
------------
- **Reproducibility**: seeds torch / numpy / python; augment pipeline is
  seeded. cuDNN deterministic is not forced.
- **Device**: CUDA if available, else CPU. AMP autocast on CUDA.
- **Sampler**: balances by class across the **combined** train set, not
  per-source. So if varroa is balanced and beeimage is skewed, the combined
  sampler corrects only the residual after concat.
- **Best metric**: balanced_accuracy. Robust to class skew on val splits.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from beevision.data.augment import classification_transforms, seed_pipeline
from beevision.data.datasets import (
    MITE_LABEL_TO_IDX,
    MiteClassificationDataset,
)
from beevision.data.paths import load_paths
from beevision.models.mite.losses import CrossEntropyMite
from beevision.models.mite.metrics import ClassificationMetricAccumulator
from beevision.models.mite.resnet50 import MiteModelConfig, build_model

LOG = logging.getLogger("train_mite")


# ---------- Config ---------------------------------------------------------


@dataclass
class TrainConfig:
    seed: int = 1337
    data_config: str = "configs/data.yaml"
    sources: tuple[str, ...] = ("varroa", "beeimage")

    # Model
    num_classes: int = 2
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout: float = 0.0

    # Train loop
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    mixed_precision: bool = True
    grad_clip: float = 1.0
    val_every: int = 1
    use_weighted_sampler: bool = True

    # Checkpoints
    checkpoint_subdir: str = "checkpoints/mite"
    best_metric: str = "balanced_accuracy"  # maximize

    # Early stop (None = disabled)
    patience: int | None = None

    extras: dict[str, Any] = field(default_factory=dict)


def load_train_config(path: str | Path) -> TrainConfig:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    model = raw.pop("model", {})
    train = raw.pop("train", {})
    ckpt = raw.pop("checkpoint", {})
    sources = raw.get("sources", ["varroa", "beeimage"])
    if isinstance(sources, str):
        sources = [sources]
    return TrainConfig(
        seed=int(raw.get("seed", 1337)),
        data_config=str(raw.get("data_config", "configs/data.yaml")),
        sources=tuple(str(s) for s in sources),
        num_classes=int(model.get("num_classes", 2)),
        backbone=str(model.get("backbone", "resnet50")),
        pretrained=bool(model.get("pretrained", True)),
        freeze_backbone=bool(model.get("freeze_backbone", False)),
        dropout=float(model.get("dropout", 0.0)),
        image_size=int(train.get("image_size", 224)),
        batch_size=int(train.get("batch_size", 64)),
        num_workers=int(train.get("num_workers", 4)),
        epochs=int(train.get("epochs", 20)),
        lr=float(train.get("lr", 3e-4)),
        weight_decay=float(train.get("weight_decay", 1e-4)),
        label_smoothing=float(train.get("label_smoothing", 0.0)),
        mixed_precision=bool(train.get("mixed_precision", True)),
        grad_clip=float(train.get("grad_clip", 1.0)),
        val_every=int(train.get("val_every", 1)),
        use_weighted_sampler=bool(train.get("use_weighted_sampler", True)),
        checkpoint_subdir=str(ckpt.get("subdir", "checkpoints/mite")),
        best_metric=str(ckpt.get("best_metric", "balanced_accuracy")),
        patience=ckpt.get("patience"),
        extras=raw,
    )


# ---------- Determinism ----------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Data -----------------------------------------------------------


def _collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    images = torch.from_numpy(np.stack([b["image"] for b in batch]))  # (B, C, H, W) float32
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)  # (B,)
    ids = [b["id"] for b in batch]
    return {"image": images, "target": targets, "id": ids}


def _build_split_datasets(
    sources: tuple[str, ...],
    split: str,
    paths,
    transform,
) -> list[MiteClassificationDataset]:
    out: list[MiteClassificationDataset] = []
    for src in sources:
        pq = paths.parquet_path(src)
        if not pq.exists():
            raise FileNotFoundError(
                f"missing {pq}; run `make ingest SRC={src}` first"
            )
        out.append(MiteClassificationDataset(pq, paths.interim, split=split, transform=transform))
    return out


def _sample_weights_for_concat(datasets: list[MiteClassificationDataset]) -> np.ndarray:
    """Per-sample weights inversely proportional to combined class counts."""
    all_labels: list[int] = []
    for ds in datasets:
        all_labels.extend(ds._df["label"].map(MITE_LABEL_TO_IDX).to_list())
    arr = np.asarray(all_labels, dtype=np.int64)
    counts = np.bincount(arr, minlength=2).astype(np.float64)
    class_w = np.where(counts > 0, counts.sum() / (2.0 * counts), 1.0)
    return class_w[arr]


def build_loaders(cfg: TrainConfig, paths) -> dict[str, DataLoader]:
    train_tf = seed_pipeline(
        classification_transforms(image_size=cfg.image_size, train=True), cfg.seed
    )
    eval_tf = seed_pipeline(
        classification_transforms(image_size=cfg.image_size, train=False), cfg.seed + 1
    )

    train_parts = _build_split_datasets(cfg.sources, "train", paths, train_tf)
    val_parts = _build_split_datasets(cfg.sources, "val", paths, eval_tf)
    test_parts = _build_split_datasets(cfg.sources, "test", paths, eval_tf)

    train_ds: ConcatDataset = ConcatDataset(train_parts)
    val_ds: ConcatDataset = ConcatDataset(val_parts)
    test_ds: ConcatDataset = ConcatDataset(test_parts)

    pin = torch.cuda.is_available()
    if cfg.use_weighted_sampler and len(train_ds) > 0:
        weights = _sample_weights_for_concat(train_parts)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
            pin_memory=pin,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
            pin_memory=pin,
            drop_last=False,
        )

    def _eval_loader(ds: ConcatDataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
            pin_memory=pin,
            drop_last=False,
        )

    return {"train": train_loader, "val": _eval_loader(val_ds), "test": _eval_loader(test_ds)}


# ---------- Train / eval loop ---------------------------------------------


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CrossEntropyMite,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float = 0.0,
    amp: bool = False,
    num_classes: int = 2,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    metric = ClassificationMetricAccumulator(num_classes=num_classes)
    totals = {"loss": 0.0, "ce": 0.0, "n": 0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            parts = criterion(logits, targets)
            loss = parts["loss"]

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        bs = images.shape[0]
        totals["loss"] += float(loss.detach()) * bs
        totals["ce"] += float(parts["ce"]) * bs
        totals["n"] += bs
        metric.update(logits.detach().float(), targets)

    n = max(totals["n"], 1)
    out: dict[str, Any] = {
        "loss": totals["loss"] / n,
        "ce": totals["ce"] / n,
    }
    out.update(metric.compute())
    return out


# ---------- Checkpointing --------------------------------------------------


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg.__dict__,
        },
        path,
    )


# ---------- Driver ---------------------------------------------------------


def main(cfg_path: str | Path) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_train_config(cfg_path)
    set_global_seed(cfg.seed)

    paths = load_paths(cfg.data_config).ensure()
    ckpt_dir = paths.processed / cfg.checkpoint_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"

    loaders = build_loaders(cfg, paths)
    LOG.info(
        "data: train=%d val=%d test=%d (sources=%s, image_size=%d)",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
        list(cfg.sources),
        cfg.image_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = MiteModelConfig(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
        dropout=cfg.dropout,
    )
    model = build_model(model_cfg).to(device)
    trainable = model.trainable_parameters()
    LOG.info(
        "model: %s | params total=%.2fM trainable=%.2fM | device=%s",
        cfg.backbone,
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for p in trainable) / 1e6,
        device,
    )

    criterion = CrossEntropyMite(label_smoothing=cfg.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best = -math.inf
    best_epoch = -1
    epochs_since_improve = 0
    t0 = time.time()
    with metrics_path.open("a") as mlog:
        for epoch in range(1, cfg.epochs + 1):
            train_m = run_epoch(
                model, loaders["train"], criterion, device,
                optimizer=optimizer, scaler=scaler,
                grad_clip=cfg.grad_clip, amp=amp,
                num_classes=cfg.num_classes,
            )
            payload: dict[str, Any] = {"epoch": epoch, "train": train_m}

            if epoch % cfg.val_every == 0 and len(loaders["val"].dataset) > 0:
                with torch.no_grad():
                    val_m = run_epoch(
                        model, loaders["val"], criterion, device,
                        amp=amp, num_classes=cfg.num_classes,
                    )
                payload["val"] = val_m
                score = val_m.get(cfg.best_metric, val_m.get("balanced_accuracy", 0.0))
                if score > best:
                    best = score
                    best_epoch = epoch
                    epochs_since_improve = 0
                    save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, val_m, cfg)
                else:
                    epochs_since_improve += 1

            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, payload, cfg)
            mlog.write(json.dumps(payload) + "\n")
            mlog.flush()
            LOG.info(
                "epoch %d/%d | train loss=%.4f acc=%.3f bal=%.3f | %s",
                epoch, cfg.epochs, train_m["loss"],
                train_m.get("accuracy", float("nan")),
                train_m.get("balanced_accuracy", float("nan")),
                (
                    f"val loss={payload['val']['loss']:.4f} "
                    f"acc={payload['val'].get('accuracy', float('nan')):.3f} "
                    f"bal={payload['val'].get('balanced_accuracy', float('nan')):.3f} "
                    f"auroc={payload['val'].get('auroc', float('nan')):.3f}"
                ) if "val" in payload else "no val this epoch",
            )

            if cfg.patience and epochs_since_improve >= cfg.patience:
                LOG.info("early stop: no val improvement for %d epochs", cfg.patience)
                break

    # Final test on best checkpoint.
    test_m: dict[str, Any] = {}
    if len(loaders["test"].dataset) > 0 and (ckpt_dir / "best.pt").exists():
        state = torch.load(ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(state["model_state"])
        with torch.no_grad():
            test_m = run_epoch(
                model, loaders["test"], criterion, device,
                amp=amp, num_classes=cfg.num_classes,
            )
        LOG.info(
            "test (best@epoch %d): loss=%.4f acc=%.3f bal=%.3f auroc=%.3f f1_mite=%.3f",
            best_epoch, test_m["loss"],
            test_m.get("accuracy", float("nan")),
            test_m.get("balanced_accuracy", float("nan")),
            test_m.get("auroc", float("nan")),
            test_m.get("f1_mite", float("nan")),
        )
        with metrics_path.open("a") as mlog:
            mlog.write(json.dumps({"test": test_m, "best_epoch": best_epoch}) + "\n")

    LOG.info(
        "done in %.1f min — best %s=%.3f at epoch %d",
        (time.time() - t0) / 60.0, cfg.best_metric, best, best_epoch,
    )
    return {"best": best, "best_epoch": best_epoch, "test": test_m}


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ResNet-50 mite classifier.")
    parser.add_argument("--config", required=True, help="Path to train_mite.yaml")
    args = parser.parse_args(argv)
    main(args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
