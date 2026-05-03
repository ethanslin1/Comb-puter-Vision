"""Cell-classifier training entry point.

Run::

    python -m beevision.models.cells.train --config configs/train_cells.yaml

Single-source training over ``deepbee_cls`` (DS-COMB-PT). The 7 classes are
imbalanced (``capped_brood``, ``nectar``, ``honey`` dominate; ``egg``,
``larva`` are rare; ``other`` is test-only and absent in train/val), so the
train loader uses a ``WeightedRandomSampler`` keyed on class index.

Checkpoints go under ``$BEEVISION_ROOT/processed/checkpoints/cells/``;
per-epoch metrics are appended as JSON lines to ``metrics.jsonl``.
``best.pt`` holds the best-by-val-macro-F1 checkpoint; ``last.pt`` holds
the most recent.

Design notes
------------
- **Reproducibility**: seeds torch / numpy / python; augment pipeline is
  seeded. cuDNN deterministic is not forced.
- **Device**: CUDA if available, else CPU. AMP autocast on CUDA.
- **Best metric**: macro_f1. Robust to the heavy class imbalance and surfaces
  per-class performance equally — what the downstream metrics module needs
  (rare-class recall matters for brood-regularity).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from beevision.data.augment import classification_transforms, seed_pipeline
from beevision.data.datasets import (
    CELL_LABEL_TO_IDX,
    CellClassificationDataset,
)
from beevision.data.paths import load_paths
from beevision.models.cells.losses import CrossEntropyCells
from beevision.models.cells.metrics import CellMetricAccumulator
from beevision.models.cells.resnet50 import CellModelConfig, build_model

LOG = logging.getLogger("train_cells")


# ---------- Config ---------------------------------------------------------


@dataclass
class TrainConfig:
    seed: int = 1337
    data_config: str = "configs/data.yaml"
    source: str = "deepbee_cls"

    # Model
    num_classes: int = 7
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
    checkpoint_subdir: str = "checkpoints/cells"
    best_metric: str = "macro_f1"  # maximize

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
        source=str(raw.get("source", "deepbee_cls")),
        num_classes=int(model.get("num_classes", 7)),
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
        checkpoint_subdir=str(ckpt.get("subdir", "checkpoints/cells")),
        best_metric=str(ckpt.get("best_metric", "macro_f1")),
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
    images = torch.from_numpy(np.stack([b["image"] for b in batch]))  # (B, C, H, W)
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)  # (B,)
    ids = [b["id"] for b in batch]
    return {"image": images, "target": targets, "id": ids}


def _sample_weights_for_dataset(
    dataset: CellClassificationDataset, num_classes: int
) -> np.ndarray:
    """Per-sample weights inversely proportional to class counts.

    Absent classes (``other`` in train) get weight 1.0 — they don't appear,
    so the value is moot but guards against div-by-zero.
    """
    labels = dataset._df["label"].map(CELL_LABEL_TO_IDX).to_numpy()
    if len(labels) == 0:
        return np.zeros(0, dtype=np.float64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_w = np.where(
            counts > 0, counts.sum() / (num_classes * counts), 1.0
        )
    return class_w[labels]


def build_loaders(cfg: TrainConfig, paths) -> dict[str, DataLoader]:
    train_tf = seed_pipeline(
        classification_transforms(image_size=cfg.image_size, train=True), cfg.seed
    )
    eval_tf = seed_pipeline(
        classification_transforms(image_size=cfg.image_size, train=False), cfg.seed + 1
    )

    pq = paths.parquet_path(cfg.source)
    if not pq.exists():
        raise FileNotFoundError(
            f"missing {pq}; run `make ingest SRC={cfg.source}` first"
        )

    train_ds = CellClassificationDataset(pq, paths.interim, split="train", transform=train_tf)
    val_ds = CellClassificationDataset(pq, paths.interim, split="val", transform=eval_tf)
    test_ds = CellClassificationDataset(pq, paths.interim, split="test", transform=eval_tf)

    pin = torch.cuda.is_available()
    if cfg.use_weighted_sampler and len(train_ds) > 0:
        weights = _sample_weights_for_dataset(train_ds, cfg.num_classes)
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

    def _eval_loader(ds: CellClassificationDataset) -> DataLoader:
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
    criterion: CrossEntropyCells,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float = 0.0,
    amp: bool = False,
    num_classes: int = 7,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    metric = CellMetricAccumulator(num_classes=num_classes)
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
        "data: train=%d val=%d test=%d (source=%s, image_size=%d)",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset),
        cfg.source,
        cfg.image_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = CellModelConfig(
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

    criterion = CrossEntropyCells(label_smoothing=cfg.label_smoothing).to(device)
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
                score = val_m.get(cfg.best_metric, val_m.get("macro_f1", 0.0))
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
                "epoch %d/%d | train loss=%.4f acc=%.3f f1=%.3f | %s",
                epoch, cfg.epochs, train_m["loss"],
                train_m.get("accuracy", float("nan")),
                train_m.get("macro_f1", float("nan")),
                (
                    f"val loss={payload['val']['loss']:.4f} "
                    f"acc={payload['val'].get('accuracy', float('nan')):.3f} "
                    f"f1={payload['val'].get('macro_f1', float('nan')):.3f} "
                    f"bal={payload['val'].get('balanced_accuracy', float('nan')):.3f}"
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
            "test (best@epoch %d): loss=%.4f acc=%.3f f1=%.3f bal=%.3f",
            best_epoch, test_m["loss"],
            test_m.get("accuracy", float("nan")),
            test_m.get("macro_f1", float("nan")),
            test_m.get("balanced_accuracy", float("nan")),
        )
        with metrics_path.open("a") as mlog:
            mlog.write(json.dumps({"test": test_m, "best_epoch": best_epoch}) + "\n")

    LOG.info(
        "done in %.1f min — best %s=%.3f at epoch %d",
        (time.time() - t0) / 60.0, cfg.best_metric, best, best_epoch,
    )
    return {"best": best, "best_epoch": best_epoch, "test": test_m}


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ResNet-50 7-class cell classifier.")
    parser.add_argument("--config", required=True, help="Path to train_cells.yaml")
    args = parser.parse_args(argv)
    main(args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
