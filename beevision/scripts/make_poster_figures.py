#!/usr/bin/env python3
"""Generate poster-ready figures + summary tables from saved training metrics.

Reads ``metrics.jsonl`` from completed segmentation and mite training runs
(default: ``$BEEVISION_ROOT/processed/checkpoints/{segmentation,mite}/``),
writes high-resolution PNGs and ``summary.json`` / ``SUMMARY.md`` under a
dedicated ``poster_results/`` directory (default: ``<repo>/poster_results``).

Optional (requires torch + checkpoints): mite ROC curve on the held-out test
set, and qualitative segmentation panels (RGB | GT | Pred | error map).

Usage (from the ``beevision/`` checkout)::

    export PYTHONPATH=src
    python scripts/make_poster_figures.py

    # custom locations
    python scripts/make_poster_figures.py \\
        --repo . \\
        --out poster_results \\
        --bevision-root /oscar/scratch/$USER/beevision \\
        --no-qualitative
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _split_epoch_rows(rows: list[dict[str, Any]]) -> tuple[list[dict], dict | None]:
    """Return (epoch_payloads, final_test_line_or_none)."""
    epochs: list[dict[str, Any]] = []
    test_line: dict[str, Any] | None = None
    for r in rows:
        if "epoch" in r:
            epochs.append(r)
        if "test" in r:
            test_line = r
    return epochs, test_line


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_segmentation_curves(epochs: list[dict], out: Path, dpi: int) -> None:
    if not epochs:
        return
    ep = [int(r["epoch"]) for r in epochs]
    tr_loss = [float(r["train"]["loss"]) for r in epochs]
    tr_iou = [float(r["train"].get("iou_comb", float("nan"))) for r in epochs]
    va_loss = [float(r["val"]["loss"]) for r in epochs if "val" in r]
    va_iou = [float(r["val"].get("iou_comb", float("nan"))) for r in epochs if "val" in r]
    ep_v = [int(r["epoch"]) for r in epochs if "val" in r]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    _style_axes(axes[0], "Segmentation — loss (train vs val)", "Epoch", "Loss (CE + Dice)")
    axes[0].plot(ep, tr_loss, color="#1f77b4", lw=2.2, label="Train")
    axes[0].plot(ep_v, va_loss, color="#ff7f0e", lw=2.2, marker="o", ms=3, label="Val")
    axes[0].legend(frameon=False, fontsize=10)

    _style_axes(axes[1], "Segmentation — IoU (comb / foreground)", "Epoch", "IoU")
    axes[1].plot(ep, tr_iou, color="#2ca02c", lw=2.2, label="Train IoU (comb)")
    axes[1].plot(ep_v, va_iou, color="#d62728", lw=2.2, marker="o", ms=3, label="Val IoU (comb)")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(frameon=False, fontsize=10)

    fig.savefig(out / "segmentation_training_curves.png", dpi=dpi)
    plt.close(fig)


def plot_segmentation_test_bars(test: dict[str, Any], out: Path, dpi: int) -> None:
    labels = ["IoU bg", "IoU comb", "Dice bg", "Dice comb", "mIoU", "mDice", "Pixel acc"]
    vals = [
        float(test["iou_per_class"][0]),
        float(test["iou_per_class"][1]),
        float(test["dice_per_class"][0]),
        float(test["dice_per_class"][1]),
        float(test["miou"]),
        float(test["mdice"]),
        float(test["pixel_acc"]),
    ]
    colors = ["#7f7f7f", "#2ca02c", "#aec7e8", "#98df8a", "#1f77b4", "#9467bd", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#cccccc", ls=":", lw=1)
    _style_axes(ax, "Segmentation — held-out test metrics (best checkpoint)", "Metric", "Score")
    for i, v in enumerate(vals):
        ax.text(i, min(v + 0.03, 0.98), f"{v:.3f}", ha="center", fontsize=9, fontweight="medium")
    fig.savefig(out / "segmentation_test_metrics_bar.png", dpi=dpi)
    plt.close(fig)


def plot_mite_curves(epochs: list[dict], out: Path, dpi: int) -> None:
    if not epochs:
        return
    ep_v = [int(r["epoch"]) for r in epochs if "val" in r]
    tr_loss = [float(r["train"]["loss"]) for r in epochs]
    va_bal = [float(r["val"]["balanced_accuracy"]) for r in epochs if "val" in r]
    va_auroc = [float(r["val"]["auroc"]) for r in epochs if "val" in r]
    ep = [int(r["epoch"]) for r in epochs]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    _style_axes(axes[0], "Mite classifier — training loss", "Epoch", "Loss (CE)")
    axes[0].plot(ep, tr_loss, color="#1f77b4", lw=2.2)
    axes[0].set_ylim(bottom=0.0)

    _style_axes(
        axes[1],
        "Mite classifier — validation (balanced acc & AUROC)",
        "Epoch",
        "Score",
    )
    axes[1].plot(ep_v, va_bal, color="#2ca02c", lw=2.2, marker="o", ms=3, label="Balanced accuracy")
    axes[1].plot(ep_v, va_auroc, color="#d62728", lw=2.2, marker="s", ms=3, label="AUROC")
    axes[1].set_ylim(0.75, 1.01)
    axes[1].legend(frameon=False, fontsize=10, loc="lower right")

    fig.savefig(out / "mite_training_curves.png", dpi=dpi)
    plt.close(fig)


def plot_mite_confusion(test: dict[str, Any], out: Path, dpi: int) -> None:
    cm = np.asarray(test["confusion"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1.0)
    cm_norm = cm / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    im0 = axes[0].imshow(cm, cmap="Blues", aspect="equal")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Pred no mite", "Pred mite"], fontsize=10)
    axes[0].set_yticklabels(["True no mite", "True mite"], fontsize=10)
    axes[0].set_title("Confusion matrix (counts)", fontsize=13, fontweight="semibold")
    for (i, j), v in np.ndenumerate(cm):
        axes[0].text(j, i, f"{int(v)}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cm_norm, vmin=0, vmax=1, cmap="Greens", aspect="equal")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Pred no mite", "Pred mite"], fontsize=10)
    axes[1].set_yticklabels(["True no mite", "True mite"], fontsize=10)
    axes[1].set_title("Row-normalized (recall view)", fontsize=13, fontweight="semibold")
    for (i, j), v in np.ndenumerate(cm_norm):
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else "black", fontsize=13)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig(out / "mite_confusion_matrix.png", dpi=dpi)
    plt.close(fig)


def plot_mite_test_bars(test: dict[str, Any], out: Path, dpi: int) -> None:
    labels = ["Accuracy", "Balanced acc.", "AUROC", "F1 (mite)"]
    vals = [
        float(test["accuracy"]),
        float(test["balanced_accuracy"]),
        float(test["auroc"]),
        float(test["f1_mite"]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"], edgecolor="white", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    _style_axes(ax, "Mite detector — held-out test (best checkpoint)", "Metric", "Score")
    for i, v in enumerate(vals):
        ax.text(i, min(v + 0.03, 0.99), f"{v:.3f}", ha="center", fontsize=10, fontweight="semibold")
    fig.savefig(out / "mite_test_metrics_bar.png", dpi=dpi)
    plt.close(fig)


def plot_mite_roc_curve(
    repo: Path,
    mite_metrics_dir: Path,
    out: Path,
    dpi: int,
) -> None:
    import torch
    from sklearn.metrics import auc, roc_curve

    from beevision.data.paths import load_paths
    from beevision.models.mite.train import build_loaders, load_train_config
    from beevision.models.mite.resnet50 import MiteModelConfig, build_model

    cfg_path = repo / "configs/train_mite.yaml"
    cfg = load_train_config(cfg_path)
    paths = load_paths(cfg.data_config).ensure()
    loaders = build_loaders(cfg, paths)
    ckpt = mite_metrics_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"mite checkpoint missing: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = MiteModelConfig(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        pretrained=False,
        freeze_backbone=cfg.freeze_backbone,
        dropout=cfg.dropout,
    )
    model = build_model(model_cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    probs_all: list[float] = []
    tgt_all: list[int] = []
    amp = cfg.mixed_precision and device.type == "cuda"
    with torch.no_grad():
        for batch in loaders["test"]:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
            p = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy().tolist()
            probs_all.extend(p)
            tgt_all.extend(targets.detach().cpu().numpy().tolist())

    fpr, tpr, _ = roc_curve(np.asarray(tgt_all), np.asarray(probs_all))
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.2, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    _style_axes(ax, "Mite classifier — ROC (held-out test)", "False positive rate", "True positive rate")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.savefig(out / "mite_roc_curve.png", dpi=dpi)
    plt.close(fig)


def plot_segmentation_qualitative(
    repo: Path,
    seg_metrics_dir: Path,
    out: Path,
    dpi: int,
    n_panels: int = 4,
) -> None:
    import torch

    from beevision.data.paths import load_paths
    from beevision.models.segmentation.train import (
        build_loaders,
        load_train_config,
        set_global_seed,
    )
    from beevision.models.segmentation.unet_dinov2 import SegModelConfig, build_model

    cfg_path = repo / "configs/train_seg.yaml"
    cfg = load_train_config(cfg_path)
    paths = load_paths(cfg.data_config).ensure()
    parquet_path = paths.parquet_path(cfg.source)
    ckpt = seg_metrics_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"seg checkpoint missing: {ckpt}")

    set_global_seed(cfg.seed)
    loaders = build_loaders(cfg, parquet_path, paths.interim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = SegModelConfig(
        num_classes=cfg.num_classes,
        encoder=cfg.encoder,
        pretrained=cfg.pretrained,
        freeze_encoder=cfg.freeze_encoder,
        decoder_channels=cfg.decoder_channels,
    )
    model = build_model(model_cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Collect first n_panels samples from test in order.
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    titles: list[str] = []
    amp = cfg.mixed_precision and device.type == "cuda"
    with torch.no_grad():
        for batch in loaders["test"]:
            im = batch["image"].to(device, non_blocking=True)
            mk = batch["mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(im)
            pr = logits.argmax(dim=1)
            for i in range(im.shape[0]):
                if len(images) >= n_panels:
                    break
                img_chw = im[i].detach().float().cpu().numpy()
                rgb = np.transpose(img_chw, (1, 2, 0))
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                images.append(rgb)
                masks.append(mk[i].detach().cpu().numpy().astype(np.uint8))
                preds.append(pr[i].detach().cpu().numpy().astype(np.uint8))
                titles.append(batch["id"][i] if i < len(batch["id"]) else f"sample_{len(images)}")
            if len(images) >= n_panels:
                break

    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(14.0, 3.2 * max(n, 1)), constrained_layout=True)
    if n == 1:
        axes = np.asarray([axes])
    col_titles = ["Input (RGB)", "Ground truth", "Prediction", "|Pred − GT|"]
    for row in range(n):
        axes[row, 0].imshow(images[row])
        axes[row, 1].imshow(masks[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(preds[row], cmap="gray", vmin=0, vmax=1)
        err = (preds[row] != masks[row]).astype(np.uint8)
        axes[row, 3].imshow(err, cmap="Reds", vmin=0, vmax=1)
        for c in range(4):
            axes[row, c].axis("off")
            if row == 0:
                axes[row, c].set_title(col_titles[c], fontsize=11, fontweight="semibold", pad=6)
        axes[row, 0].text(
            0.02,
            0.98,
            titles[row],
            transform=axes[row, 0].transAxes,
            fontsize=8,
            color="white",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.45),
        )
    fig.suptitle(
        "Segmentation — qualitative test-set examples (DINOv2-UNet, best ckpt)",
        fontsize=13,
        fontweight="semibold",
        y=1.01,
    )
    fig.savefig(out / "segmentation_qualitative_test.png", dpi=dpi)
    plt.close(fig)


def _write_summary_md(
    out: Path,
    seg_test: dict[str, Any] | None,
    mite_test: dict[str, Any] | None,
    seg_best_ep: int | None,
    mite_best_ep: int | None,
) -> None:
    lines = [
        "# BeeVision — poster results summary",
        "",
        "Auto-generated; re-run `make poster-results` after new training runs.",
        "",
        "## Segmentation (DS-COMB / `deepbee_seg`, DINOv2 ViT-S + UNet)",
        "",
    ]
    if seg_test is not None:
        lines += [
            f"- **Best checkpoint (val IoU comb):** epoch {seg_best_ep}",
            f"- **Test IoU (comb):** {seg_test['iou_comb']:.4f}",
            f"- **Test mIoU:** {seg_test['miou']:.4f}",
            f"- **Test Dice (comb):** {seg_test['dice_comb']:.4f}",
            f"- **Test pixel accuracy:** {seg_test['pixel_acc']:.4f}",
            "",
        ]
    else:
        lines.append("_No segmentation test metrics found in metrics.jsonl._\n")

    lines += [
        "## Mite detector (Varroa + BeeImage, ResNet-50)",
        "",
    ]
    if mite_test is not None:
        lines += [
            f"- **Best checkpoint (val balanced accuracy):** epoch {mite_best_ep}",
            f"- **Test accuracy:** {mite_test['accuracy']:.4f}",
            f"- **Test balanced accuracy:** {mite_test['balanced_accuracy']:.4f}",
            f"- **Test AUROC:** {mite_test['auroc']:.4f}",
            f"- **Test F1 (mite):** {mite_test['f1_mite']:.4f}",
            "",
        ]
    else:
        lines.append("_No mite test metrics found in metrics.jsonl._\n")

    lines += [
        "## Figure files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `segmentation_training_curves.png` | Train/val loss & IoU (comb) |",
        "| `segmentation_test_metrics_bar.png` | Held-out test scalar metrics |",
        "| `mite_training_curves.png` | Train loss; val balanced acc & AUROC |",
        "| `mite_confusion_matrix.png` | Confusion counts + row-normalized |",
        "| `mite_test_metrics_bar.png` | Held-out test headline metrics |",
        "| `mite_roc_curve.png` | ROC on test (re-eval from `best.pt`) |",
        "| `segmentation_qualitative_test.png` | RGB / GT / pred / error |",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _discover_bevision_root(repo: Path, user: str, explicit: Path | None) -> Path:
    """Resolve the data root that actually contains ``processed/checkpoints``.

    Training writes ``metrics.jsonl`` next to ``best.pt``. On some checkouts
    ``$BEEVISION_ROOT`` points at a symlinked ``beevision/data/`` tree; we
    therefore probe a few candidates and pick the first where checkpoint
    metrics exist.
    """
    if explicit is not None:
        return explicit.resolve()
    roots: list[Path] = []
    env = os.environ.get("BEEVISION_ROOT")
    if env:
        roots.append(Path(env).resolve())
    roots.append(Path(f"/oscar/scratch/{user}/beevision").resolve())
    roots.append((repo / "data").resolve())
    seen: set[str] = set()
    for r in roots:
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        seg_m = r / "processed/checkpoints/segmentation/metrics.jsonl"
        mite_m = r / "processed/checkpoints/mite/metrics.jsonl"
        if seg_m.is_file() or mite_m.is_file():
            return r
    return roots[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build poster figures from metrics.jsonl + checkpoints.")
    ap.add_argument("--repo", type=Path, default=_repo_root(), help="BeeVision repo root (default: auto)")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: <repo>/poster_results)")
    ap.add_argument(
        "--bevision-root",
        type=Path,
        default=None,
        help="BEEVISION_ROOT (default: env or /oscar/scratch/$USER/beevision)",
    )
    ap.add_argument("--dpi", type=int, default=220, help="PNG resolution")
    ap.add_argument("--no-roc", action="store_true", help="Skip mite ROC (needs torch + best.pt)")
    ap.add_argument("--no-qualitative", action="store_true", help="Skip seg qualitative panel")
    args = ap.parse_args()

    repo = args.repo.resolve()
    out = (args.out or (repo / "poster_results")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    user = os.environ.get("USER", "user")
    bev = _discover_bevision_root(repo, user, args.bevision_root)

    seg_jsonl = bev / "processed/checkpoints/segmentation/metrics.jsonl"
    mite_jsonl = bev / "processed/checkpoints/mite/metrics.jsonl"

    summary: dict[str, Any] = {"bevision_root": str(bev), "repo": str(repo)}

    prev_cwd = os.getcwd()
    try:
        os.chdir(repo)
        if str(repo / "src") not in sys.path:
            sys.path.insert(0, str(repo / "src"))

        # --- Segmentation ---
        if seg_jsonl.exists():
            s_rows = _read_jsonl(seg_jsonl)
            s_epochs, s_test_line = _split_epoch_rows(s_rows)
            plot_segmentation_curves(s_epochs, out, args.dpi)
            if s_test_line and "test" in s_test_line:
                st = s_test_line["test"]
                plot_segmentation_test_bars(st, out, args.dpi)
                summary["segmentation"] = {
                    "best_epoch": s_test_line.get("best_epoch"),
                    "test": {
                        k: float(v)
                        for k, v in st.items()
                        if isinstance(v, (int, float))
                    },
                }
                if not args.no_qualitative:
                    try:
                        plot_segmentation_qualitative(repo, seg_jsonl.parent, out, args.dpi, n_panels=4)
                        summary["segmentation"]["qualitative_png"] = "segmentation_qualitative_test.png"
                    except Exception as e:  # noqa: BLE001
                        summary["segmentation"]["qualitative_error"] = str(e)
            else:
                summary["segmentation"] = {"error": "no test line in metrics.jsonl"}
        else:
            summary["segmentation"] = {"error": f"missing {seg_jsonl}"}

        # --- Mite ---
        if mite_jsonl.exists():
            m_rows = _read_jsonl(mite_jsonl)
            m_epochs, m_test_line = _split_epoch_rows(m_rows)
            plot_mite_curves(m_epochs, out, args.dpi)
            if m_test_line and "test" in m_test_line:
                mt = m_test_line["test"]
                plot_mite_confusion(mt, out, args.dpi)
                plot_mite_test_bars(mt, out, args.dpi)
                summary["mite"] = {
                    "best_epoch": m_test_line.get("best_epoch"),
                    "test": {
                        k: float(v)
                        for k, v in mt.items()
                        if isinstance(v, (int, float))
                    },
                }
                if not args.no_roc:
                    try:
                        plot_mite_roc_curve(repo, mite_jsonl.parent, out, args.dpi)
                        summary["mite"]["roc_png"] = "mite_roc_curve.png"
                    except Exception as e:  # noqa: BLE001
                        summary["mite"]["roc_error"] = str(e)
            else:
                summary["mite"] = {"error": "no test line in metrics.jsonl"}
        else:
            summary["mite"] = {"error": f"missing {mite_jsonl}"}
    finally:
        os.chdir(prev_cwd)

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    seg_te = (
        summary.get("segmentation", {}).get("test")
        if isinstance(summary.get("segmentation"), dict)
        else None
    )
    mite_te = (
        summary.get("mite", {}).get("test")
        if isinstance(summary.get("mite"), dict)
        else None
    )
    _write_summary_md(
        out,
        seg_te,
        mite_te,
        summary.get("segmentation", {}).get("best_epoch")
        if isinstance(summary.get("segmentation"), dict)
        else None,
        summary.get("mite", {}).get("best_epoch")
        if isinstance(summary.get("mite"), dict)
        else None,
    )

    print(f"Wrote poster assets → {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
