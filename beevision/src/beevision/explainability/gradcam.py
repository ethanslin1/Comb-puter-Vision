"""Grad-CAM (Selvaraju et al. 2017) implemented with PyTorch hooks.

Pure-torch implementation — no ``pytorch_grad_cam`` dependency. Algorithm:

    1. Register a forward hook that captures activations of a chosen layer.
    2. Register a backward hook that captures gradients flowing back to that
       layer.
    3. Forward the input through the model, then backprop ``logits[:, k]``
       (the target class) for each sample in the batch.
    4. Per channel, take the spatial mean of the gradients — these are the
       channel weights ``α_c``.
    5. Compute the weighted sum ``Σ_c α_c · A_c`` of activations across
       channels, ReLU it, and bilinear-upsample to input resolution.
    6. Normalize to [0, 1] for visualization.

For the BeeVision ResNet-50 classifiers, the canonical target layer is
``model.backbone.layer4[-1]`` (final residual block, 7×7 feature map at
224×224 input). We expose a helper ``find_default_target_layer`` that
locates it for both ``ResNet50Mite`` and ``ResNet50Cells``.

Memory note: hooks store batched activations and grads on the device.
Run inside ``torch.cuda.amp.autocast(False)`` (default) — Grad-CAM is
not numerically delicate but float32 makes the small-batch math simpler.

The ``GradCAM`` class is a context manager so hooks are guaranteed to be
removed even if the forward/backward raises::

    with GradCAM(model, layer) as cam:
        heatmaps = cam(images, target_class=1)

If you only need a one-off, use ``gradcam(model, layer, image, target=k)``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("explainability.gradcam")


class GradCAM:
    """Grad-CAM with a chosen feature layer.

    The model is *not* put in any particular train/eval mode by this class —
    callers should set ``.eval()`` before computing CAMs. Disable autocast
    explicitly if your training stack uses it.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> "GradCAM":
        # Forward hook captures the layer's output.
        def _fwd_hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            self._activations = out.detach()

        # Backward hook captures grads w.r.t. the layer's output.
        # ``register_full_backward_hook`` is the modern stable API.
        def _bwd_hook(_mod: nn.Module, _grad_in: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
            self._gradients = grad_out[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(_fwd_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(_bwd_hook))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._activations = None
        self._gradients = None
        
    def __call__(
        self,
        images: torch.Tensor,
        target_class: int | list[int] | None = None,
    ) -> np.ndarray:
        """Compute CAMs for a batch of images.

        ``target_class`` can be:
        - ``None`` → use argmax of logits per sample (the model's prediction).
        - ``int`` → same target class for the whole batch.
        - ``list[int]`` → per-sample target class (must have ``len == batch``).

        Returns: ``(B, H, W)`` float32 array in [0, 1].
        """
        if not self._handles:
            raise RuntimeError("GradCAM must be used inside a ``with`` block")
        if images.ndim != 4:
            raise ValueError(f"images must be (B, C, H, W); got shape {tuple(images.shape)}")

        B, _, H, W = images.shape
        # Forward pass.
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        if logits.ndim != 2:
            raise ValueError(
                f"GradCAM expects classifier logits (B, C); got shape {tuple(logits.shape)}"
            )

        # Resolve per-sample target classes.
        if target_class is None:
            targets = logits.argmax(dim=1).tolist()
        elif isinstance(target_class, int):
            targets = [target_class] * B
        else:
            if len(target_class) != B:
                raise ValueError(
                    f"target_class list length {len(target_class)} != batch {B}"
                )
            targets = list(target_class)

        # Build a one-hot signal and backprop.
        one_hot = torch.zeros_like(logits)
        one_hot[torch.arange(B), targets] = 1.0
        # Sum so we backprop a scalar; per-sample grads are still distinct
        # because gradients flow only along each sample's selected logit.
        loss = (one_hot * logits).sum()
        loss.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "hooks did not fire; check that target_layer is part of model's forward"
            )

        acts = self._activations            # (B, C, H', W')
        grads = self._gradients             # (B, C, H', W')

        # Channel weights: spatial mean of grads.
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (B, 1, H', W')
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)                # (B, H, W)

        # Per-sample min-max normalize to [0, 1]; flat-zero CAMs stay zero.
        flat = cam.view(B, -1)
        cmin = flat.min(dim=1, keepdim=True).values
        cmax = flat.max(dim=1, keepdim=True).values
        denom = (cmax - cmin).clamp_min(1e-6)
        cam = ((flat - cmin) / denom).view(B, H, W)
        cam = cam.clamp(0.0, 1.0)
        return cam.detach().cpu().numpy().astype(np.float32)


def find_default_target_layer(model: nn.Module) -> nn.Module:
    """Return the canonical Grad-CAM layer for a BeeVision classifier.

    For both ``ResNet50Mite`` and ``ResNet50Cells`` this is
    ``model.backbone.layer4[-1]`` — the last residual block of the last
    stage, which produces a 7×7 feature map at 224×224 input. Falls back
    to scanning for a ``layer4`` attribute for unknown architectures.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError(
            "expected model.backbone (e.g. torchvision ResNet); "
            "pass target_layer explicitly for custom architectures"
        )
    layer4 = getattr(backbone, "layer4", None)
    if layer4 is None or len(list(layer4.children())) == 0:
        raise AttributeError(
            "model.backbone has no layer4; pass target_layer explicitly"
        )
    return layer4[-1]


def gradcam(
    model: nn.Module,
    image: torch.Tensor,
    *,
    target_class: int | list[int] | None = None,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """One-shot helper. Adds a batch dim if ``image`` is (C, H, W).

    Returns ``(H, W)`` for single images, ``(B, H, W)`` for batches.
    """
    single = image.ndim == 3
    if single:
        image = image.unsqueeze(0)
    if target_layer is None:
        target_layer = find_default_target_layer(model)
    with GradCAM(model, target_layer) as cam:
        out = cam(image, target_class=target_class)
    return out[0] if single else out
