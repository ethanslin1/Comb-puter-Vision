"""Segmentation training entry point.

Run:

    python -m beevision.models.segmentation.train --config configs/train_seg.yaml

The YAML config points at ``configs/data.yaml`` for paths and at ``deepbee_seg``
for data. Checkpoints go under ``$BEEVISION_ROOT/processed/checkpoints/segmentation/``;
per-epoch metrics are appended as JSON lines to the same directory's
``metrics.jsonl``. ``best.pt`` holds the best-by-val-IoU checkpoint; ``last.pt``
holds the most recent.

Design notes
------------
- **Reproducibility**: seeds torch / numpy / python; `torch.backends.cudnn.deterministic`
  is not forced (too slow for training), but the seeded data pipeline keeps augment
  draws reproducible. Final metric still varies run-to-run due to cudnn kernels.
- **Device**: CUDA if available, else CPU. Mixed precision (autocast) is enabled
  on CUDA by default.
- **Encoder freezing**: ``model.cfg.freeze_encoder`` is the single source of truth;
  the optimizer sees only ``model.trainable_parameters()``.
- **Why no lr scheduler yet**: with 42 train frames and ~50 epochs, a cosine
  schedule is overkill. If val IoU plateaus, add one.
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
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from beevision.data.augment import seed_pipeline, segmentation_transforms
from beevision.data.datasets import FrameSegmentationDataset
from beevision.data.paths import load_paths
from beevision.models.segmentation.losses import CEDiceLoss
from beevision.models.segmentation.metrics import SegMetricAccumulator
from beevision.models.segmentation.unet_dinov2 import (
    DINOV2_PATCH,
    SegModelConfig,
    build_model,
)

LOG = logging.getLogger("train_seg")


# ---------- Config ---------------------------------------------------------


@dataclass
class TrainConfig:
    seed: int = 1337
    data_config: str = "configs/data.yaml"
    source: str = "deepbee_seg"

    # Model
    num_classes: int = 2
    encoder: str = "dinov2_vits14"
    pretrained: bool = True
    freeze_encoder: bool = True
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32)

    # Train loop
    image_size: int = 518  # 37 * 14; divisible by DINOv2 patch
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.01
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    mixed_precision: bool = True
    grad_clip: float = 1.0
    val_every: int = 1

    # Checkpoints
    checkpoint_subdir: str = "checkpoints/segmentation"
    best_metric: str = "iou_comb"  # maximize

    # Early stop (None = disabled)
    patience: int | None = None

    extras: dict[str, Any] = field(default_factory=dict)


def load_train_config(path: str | Path) -> TrainConfig:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    model = raw.pop("model", {})
    train = raw.pop("train", {})
    ckpt = raw.pop("checkpoint", {})
    return TrainConfig(
        seed=int(raw.get("seed", 1337)),
        data_config=str(raw.get("data_config", "configs/data.yaml")),
        source=str(raw.get("source", "deepbee_seg")),
        num_classes=int(model.get("num_classes", 2)),
        encoder=str(model.get("encoder", "dinov2_vits14")),
        pretrained=bool(model.get("pretrained", True)),
        freeze_encoder=bool(model.get("freeze_encoder", True)),
        decoder_channels=tuple(model.get("decoder_channels", (256, 128, 64, 32))),
        image_size=int(train.get("image_size", 518)),
        batch_size=int(train.get("batch_size", 4)),
        num_workers=int(train.get("num_workers", 2)),
        epochs=int(train.get("epochs", 50)),
        lr=float(train.get("lr", 3e-4)),
        weight_decay=float(train.get("weight_decay", 0.01)),
        ce_weight=float(train.get("ce_weight", 1.0)),
        dice_weight=float(train.get("dice_weight", 1.0)),
        mixed_precision=bool(train.get("mixed_precision", True)),
        grad_clip=float(train.get("grad_clip", 1.0)),
        val_every=int(train.get("val_every", 1)),
        checkpoint_subdir=str(ckpt.get("subdir", "checkpoints/segmentation")),
        best_metric=str(ckpt.get("best_metric", "iou_comb")),
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
    masks = torch.from_numpy(np.stack([b["mask"] for b in batch]))  # (B, H, W) int64
    ids = [b["id"] for b in batch]
    return {"image": images, "mask": masks, "id": ids}


def build_loaders(
    cfg: TrainConfig, parquet_path: Path, interim_dir: Path
) -> dict[str, DataLoader]:
    if cfg.image_size % DINOV2_PATCH:
        raise ValueError(
            f"image_size={cfg.image_size} is not divisible by DINOv2 patch={DINOV2_PATCH}; "
            f"try {(cfg.image_size // DINOV2_PATCH) * DINOV2_PATCH} or "
            f"{(cfg.image_size // DINOV2_PATCH + 1) * DINOV2_PATCH}"
        )
    train_tf = seed_pipeline(
        segmentation_transforms(image_size=cfg.image_size, train=True), cfg.seed
    )
    eval_tf = seed_pipeline(
        segmentation_transforms(image_size=cfg.image_size, train=False), cfg.seed + 1
    )

    def _make(split: str, shuffle: bool, transform) -> DataLoader:
        ds = FrameSegmentationDataset(parquet_path, interim_dir, split=split, transform=transform)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return {
        "train": _make("train", shuffle=True, transform=train_tf),
        "val": _make("val", shuffle=False, transform=eval_tf),
        "test": _make("test", shuffle=False, transform=eval_tf),
    }


# ---------- Train / eval loops --------------------------------------------


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CEDiceLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float = 0.0,
    amp: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    metric = SegMetricAccumulator(num_classes=int(getattr(model, "cfg", SegModelConfig()).num_classes))
    totals = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "n": 0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            parts = criterion(logits, masks)
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
        totals["dice"] += float(parts["dice"]) * bs
        totals["n"] += bs
        metric.update(logits.detach().float(), masks)

    n = max(totals["n"], 1)
    out = {
        "loss": totals["loss"] / n,
        "ce": totals["ce"] / n,
        "dice_loss": totals["dice"] / n,
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
    parquet_path = paths.parquet_path(cfg.source)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing {parquet_path}; run `make ingest SRC={cfg.source}` first"
        )
    ckpt_dir = paths.processed / cfg.checkpoint_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"

    loaders = build_loaders(cfg, parquet_path, paths.interim)
    LOG.info(
        "data: train=%d val=%d test=%d (image_size=%d)",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
        cfg.image_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = SegModelConfig(
        num_classes=cfg.num_classes,
        encoder=cfg.encoder,
        pretrained=cfg.pretrained,
        freeze_encoder=cfg.freeze_encoder,
        decoder_channels=cfg.decoder_channels,
    )
    model = build_model(model_cfg).to(device)
    trainable = model.trainable_parameters()
    LOG.info(
        "model: %s | params total=%.2fM trainable=%.2fM | device=%s",
        cfg.encoder,
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for p in trainable) / 1e6,
        device,
    )

    criterion = CEDiceLoss(ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight).to(device)
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
            )
            payload: dict[str, Any] = {"epoch": epoch, "train": train_m}

            if epoch % cfg.val_every == 0 and len(loaders["val"].dataset) > 0:
                with torch.no_grad():
                    val_m = run_epoch(model, loaders["val"], criterion, device, amp=amp)
                payload["val"] = val_m
                score = val_m.get(cfg.best_metric, val_m.get("miou", 0.0))
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
                "epoch %d/%d | train loss=%.4f iou_comb=%.3f | %s",
                epoch, cfg.epochs, train_m["loss"],
                train_m.get("iou_comb", float("nan")),
                (
                    f"val loss={payload['val']['loss']:.4f} "
                    f"iou_comb={payload['val'].get('iou_comb', float('nan')):.3f}"
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
            test_m = run_epoch(model, loaders["test"], criterion, device, amp=amp)
        LOG.info(
            "test (best@epoch %d): loss=%.4f iou_comb=%.3f miou=%.3f",
            best_epoch, test_m["loss"],
            test_m.get("iou_comb", float("nan")),
            test_m.get("miou", float("nan")),
        )
        with metrics_path.open("a") as mlog:
            mlog.write(json.dumps({"test": test_m, "best_epoch": best_epoch}) + "\n")

    LOG.info(
        "done in %.1f min — best %s=%.3f at epoch %d",
        (time.time() - t0) / 60.0, cfg.best_metric, best, best_epoch,
    )
    return {"best": best, "best_epoch": best_epoch, "test": test_m}


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train DINOv2-UNet segmentation.")
    parser.add_argument("--config", required=True, help="Path to train_seg.yaml")
    args = parser.parse_args(argv)
    main(args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
