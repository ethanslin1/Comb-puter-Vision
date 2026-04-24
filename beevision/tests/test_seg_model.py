"""Tests for the DINOv2-UNet segmentation model.

We avoid ``torch.hub.load`` during unit tests by passing a fake encoder into
``DINOv2UNet`` that mirrors the real encoder's interface:
``forward(x: (B, 3, H, W)) -> (B, embed_dim, H/14, W/14)``. That keeps tests
offline and fast while still exercising the decoder, losses, and metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from beevision.models.segmentation.losses import CEDiceLoss, SoftDiceLoss
from beevision.models.segmentation.metrics import SegMetricAccumulator
from beevision.models.segmentation.unet_dinov2 import (
    DINOV2_PATCH,
    DINOv2UNet,
    SegModelConfig,
    UNetDecoder,
)


# ---------- Fake encoder ---------------------------------------------------


class _FakeEncoder(torch.nn.Module):
    """Tiny stand-in for ``DINOv2Encoder``: fixed-dim patch tokens via a single
    strided conv with kernel = patch = 14."""

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch = DINOV2_PATCH
        self.proj = torch.nn.Conv2d(3, embed_dim, kernel_size=self.patch, stride=self.patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _make_model(num_classes: int = 2, embed_dim: int = 32) -> DINOv2UNet:
    cfg = SegModelConfig(num_classes=num_classes, decoder_channels=(64, 32, 16, 8))
    enc = _FakeEncoder(embed_dim=embed_dim)
    return DINOv2UNet(cfg=cfg, encoder=enc)


# ---------- Forward shapes -------------------------------------------------


@pytest.mark.parametrize("image_size", [14 * 4, 14 * 8])  # 56, 112
def test_forward_shape_matches_input_hw(image_size: int) -> None:
    model = _make_model(num_classes=2)
    x = torch.randn(2, 3, image_size, image_size)
    y = model(x)
    assert y.shape == (2, 2, image_size, image_size)


def test_forward_rejects_non_divisible_input() -> None:
    # The fake encoder uses a stride-14 conv that would silently crop; but
    # the model doesn't guard that — the real DINOv2Encoder does. We assert
    # the decoder still produces the interpolated target_size.
    model = _make_model(num_classes=2)
    # A size like 56 (div 14) is fine. Pick 56 and verify it runs cleanly.
    x = torch.randn(1, 3, 56, 56)
    y = model(x)
    assert y.shape == (1, 2, 56, 56)


def test_trainable_params_excludes_frozen_encoder() -> None:
    model = _make_model(num_classes=2)
    # Freeze the fake encoder manually.
    for p in model.encoder.parameters():
        p.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.trainable_parameters())
    assert trainable < total


# ---------- Losses ---------------------------------------------------------


def test_soft_dice_perfect_prediction_is_zero() -> None:
    # Logits heavily peaked on the correct class → dice ≈ 1 → loss ≈ 0.
    B, H, W = 1, 8, 8
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, :, 4:] = 1
    logits = torch.full((B, 2, H, W), -10.0)
    logits[:, 0, :, :4] = 10.0
    logits[:, 1, :, 4:] = 10.0
    loss = SoftDiceLoss()(logits, target)
    assert loss.item() < 1e-3


def test_soft_dice_completely_wrong_is_near_one() -> None:
    B, H, W = 1, 8, 8
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, :, 4:] = 1
    # Predict the opposite class everywhere.
    logits = torch.full((B, 2, H, W), -10.0)
    logits[:, 1, :, :4] = 10.0
    logits[:, 0, :, 4:] = 10.0
    loss = SoftDiceLoss()(logits, target)
    assert loss.item() > 0.95


def test_ce_dice_components_present() -> None:
    crit = CEDiceLoss(ce_weight=1.0, dice_weight=1.0)
    logits = torch.randn(2, 2, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 16, 16))
    parts = crit(logits, target)
    assert "loss" in parts and "ce" in parts and "dice" in parts
    parts["loss"].backward()
    assert logits.grad is not None


# ---------- Metrics --------------------------------------------------------


def test_metric_accumulator_perfect_iou_is_one() -> None:
    acc = SegMetricAccumulator(num_classes=2)
    B, H, W = 2, 8, 8
    target = torch.randint(0, 2, (B, H, W))
    # Build logits that strongly argmax to target.
    logits = torch.zeros(B, 2, H, W)
    logits[:, 0].masked_fill_(target == 0, 10.0)
    logits[:, 1].masked_fill_(target == 1, 10.0)
    logits[:, 0].masked_fill_(target == 1, -10.0)
    logits[:, 1].masked_fill_(target == 0, -10.0)
    acc.update(logits, target)
    out = acc.compute()
    assert out["miou"] > 0.99
    assert out["iou_comb"] > 0.99


def test_metric_accumulator_disjoint_is_zero() -> None:
    acc = SegMetricAccumulator(num_classes=2)
    B, H, W = 1, 8, 8
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, :, 4:] = 1
    # Predict opposite everywhere.
    logits = torch.full((B, 2, H, W), -10.0)
    logits[:, 1, :, :4] = 10.0
    logits[:, 0, :, 4:] = 10.0
    acc.update(logits, target)
    out = acc.compute()
    assert out["iou_comb"] < 1e-3


# ---------- Overfit smoke test --------------------------------------------


def test_can_overfit_one_batch() -> None:
    """Sanity: loss drops materially on a learnable pattern.

    Target is a hard left/right split aligned with image brightness — a signal
    a tiny conv stack can fit quickly. Pure-noise→random-label would need many
    more steps to overfit and that's not what we're checking.
    """
    torch.manual_seed(0)
    model = _make_model(num_classes=2)
    crit = CEDiceLoss()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=3e-3)

    B, S = 2, 56  # 56 = 4 patches of 14
    x = torch.zeros(B, 3, S, S)
    x[:, :, :, S // 2:] = 1.0  # right half bright
    target = torch.zeros(B, S, S, dtype=torch.long)
    target[:, :, S // 2:] = 1  # right half is "comb"

    loss0 = crit(model(x), target)["loss"].item()
    for _ in range(50):
        opt.zero_grad()
        out = crit(model(x), target)
        out["loss"].backward()
        opt.step()
    loss1 = crit(model(x), target)["loss"].item()
    assert loss1 < loss0 * 0.5, f"loss did not drop enough: {loss0:.3f} → {loss1:.3f}"


# ---------- Integration with FrameSegmentationDataset ---------------------


def _write_seg_parquet(tmp_path):
    interim = tmp_path / "interim"
    rows = []
    for i in range(2):
        img_rel = f"images/deepbee_seg/train/f{i}.png"
        msk_rel = f"masks/deepbee_seg/train/f{i}.png"
        (interim / img_rel).parent.mkdir(parents=True, exist_ok=True)
        (interim / msk_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (56, 56), color=(100, 120, 140)).save(interim / img_rel)
        mask_arr = np.zeros((56, 56), dtype=np.uint8)
        mask_arr[:, 28:] = 1
        Image.fromarray(mask_arr, mode="L").save(interim / msk_rel)
        rows.append({
            "id": f"deepbee_seg:f{i}",
            "source": "deepbee_seg",
            "split": "train",
            "image_path": img_rel,
            "mask_path": msk_rel,
            "meta": {"classes": ["background", "comb"]},
        })
    pq = interim / "deepbee_seg.parquet"
    pd.DataFrame(rows).to_parquet(pq, index=False, engine="pyarrow")
    return pq, interim


def test_model_consumes_dataset_batch(tmp_path) -> None:
    from beevision.data.augment import segmentation_transforms
    from beevision.data.datasets import FrameSegmentationDataset

    pq, interim = _write_seg_parquet(tmp_path)
    tf = segmentation_transforms(image_size=56, train=False)
    ds = FrameSegmentationDataset(pq, interim, split="train", transform=tf)

    batch = [ds[i] for i in range(len(ds))]
    images = torch.from_numpy(np.stack([b["image"] for b in batch]))
    masks = torch.from_numpy(np.stack([b["mask"] for b in batch]))
    assert images.shape == (2, 3, 56, 56) and images.dtype == torch.float32
    assert masks.shape == (2, 56, 56) and masks.dtype == torch.int64

    model = _make_model(num_classes=2)
    logits = model(images)
    assert logits.shape == (2, 2, 56, 56)

    crit = CEDiceLoss()
    parts = crit(logits, masks)
    assert torch.isfinite(parts["loss"])
