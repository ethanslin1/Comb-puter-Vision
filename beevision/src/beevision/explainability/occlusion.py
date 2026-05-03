"""Occlusion-sensitivity maps (Zeiler & Fergus 2014).

Slide a square mask across the input image; for each position, measure
how the target-class **logit** (pre-softmax score) drops compared to the
unoccluded image. The resulting heatmap shows which regions, when
removed, most hurt the prediction — a complementary view to Grad-CAM
that is fully model-agnostic and has no dependency on backprop or layer
choice.

Why logits, not probabilities: a confident classifier saturates softmax
to 1.0 for its predicted class, so post-softmax drops fall below
floating-point precision and the heatmap collapses to all zeros. Logits
have no upper bound, so even a saturated model's per-position
sensitivity remains visible. (This matches Captum and most modern
attribution libraries.)

Why ship this alongside Grad-CAM:
- **No external dependencies** (Grad-CAM also has none, but LIME does).
  Useful as a working "LIME-style" attribution out of the box.
- **Model-agnostic**: works on any classifier without picking a target
  layer or doing backward passes; useful for comparing CAM vs occlusion
  for the same prediction.
- **Slower** than Grad-CAM (one forward pass per occlusion position) but
  embarrassingly batchable — we batch all occlusion positions per image.

API mirrors ``gradcam``: takes ``(C, H, W)`` or ``(B, C, H, W)``, returns
``(H, W)`` or ``(B, H, W)`` heatmap in [0, 1] where 1 = highest sensitivity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

LOG = logging.getLogger("explainability.occlusion")


@dataclass
class OcclusionConfig:
    patch_size: int = 32           # square mask edge in pixels
    stride: int = 16               # slide step in pixels
    fill_value: float = 0.0        # value to fill the masked patch
    forward_batch: int = 32        # forward passes per chunk


def _build_mask_grid(H: int, W: int, patch: int, stride: int) -> tuple[list[tuple[int, int]], int, int]:
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    if ys[-1] + patch < H:
        ys.append(H - patch)
    if xs[-1] + patch < W:
        xs.append(W - patch)
    return [(y, x) for y in ys for x in xs], len(ys), len(xs)


def occlusion_map(
    model: nn.Module,
    images: torch.Tensor,
    *,
    target_class: int | list[int] | None = None,
    cfg: OcclusionConfig | None = None,
) -> np.ndarray:
    """Compute occlusion-sensitivity heatmaps for a batch of images.

    Returns a ``(B, H, W)`` float32 array in [0, 1]. Each pixel's value is
    the (clipped, normalized) drop in target-class probability when the
    patch covering that pixel is occluded — high values mean the model
    relies on this region.
    """
    cfg = cfg or OcclusionConfig()
    if images.ndim != 4:
        raise ValueError(f"images must be (B, C, H, W); got {tuple(images.shape)}")

    B, C, H, W = images.shape
    device = images.device

    # Baseline logits for the target class per image.
    with torch.no_grad():
        base_logits = model(images)
        if target_class is None:
            targets = base_logits.argmax(dim=1).tolist()
        elif isinstance(target_class, int):
            targets = [target_class] * B
        else:
            if len(target_class) != B:
                raise ValueError(
                    f"target_class list length {len(target_class)} != batch {B}"
                )
            targets = list(target_class)
        idx = torch.arange(B, device=device)
        base = base_logits[idx, torch.tensor(targets, device=device)]  # (B,)

    positions, ny, nx = _build_mask_grid(H, W, cfg.patch_size, cfg.stride)
    n_pos = len(positions)
    if n_pos == 0:
        return np.zeros((B, H, W), dtype=np.float32)

    # Per (image, position) drop matrix; we'll splat back to a heatmap.
    drops = torch.zeros(B, n_pos, device=device)

    for b in range(B):
        img = images[b:b + 1]
        target = targets[b]
        # Build a stack of occluded copies.
        for chunk_start in range(0, n_pos, cfg.forward_batch):
            chunk = positions[chunk_start:chunk_start + cfg.forward_batch]
            stack = img.repeat(len(chunk), 1, 1, 1).clone()
            for k, (y, x) in enumerate(chunk):
                stack[k, :, y:y + cfg.patch_size, x:x + cfg.patch_size] = cfg.fill_value
            with torch.no_grad():
                logits = model(stack)[:, target]
            drops[b, chunk_start:chunk_start + len(chunk)] = (base[b] - logits).clamp_min(0)

    # Splat drops onto (B, H, W) heatmap; pixels covered by multiple patches
    # are averaged (count grid).
    heat = torch.zeros(B, H, W, device=device)
    cover = torch.zeros(H, W, device=device)
    for k, (y, x) in enumerate(positions):
        heat[:, y:y + cfg.patch_size, x:x + cfg.patch_size] += drops[:, k:k + 1, None]
        cover[y:y + cfg.patch_size, x:x + cfg.patch_size] += 1.0
    heat = heat / cover.clamp_min(1.0)

    # Normalize per image to [0, 1].
    flat = heat.view(B, -1)
    cmin = flat.min(dim=1, keepdim=True).values
    cmax = flat.max(dim=1, keepdim=True).values
    denom = (cmax - cmin).clamp_min(1e-6)
    out = ((flat - cmin) / denom).view(B, H, W).clamp(0.0, 1.0)
    return out.detach().cpu().numpy().astype(np.float32)


def occlusion(
    model: nn.Module,
    image: torch.Tensor,
    *,
    target_class: int | list[int] | None = None,
    cfg: OcclusionConfig | None = None,
) -> np.ndarray:
    """One-shot helper for a single image (C, H, W) or batch (B, C, H, W)."""
    single = image.ndim == 3
    if single:
        image = image.unsqueeze(0)
    out = occlusion_map(model, image, target_class=target_class, cfg=cfg)
    return out[0] if single else out
