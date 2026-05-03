"""LIME (Ribeiro et al. 2016) wrapper for image classifiers.

Thin shim over the third-party ``lime`` package — that library handles the
expensive bits (SLIC superpixel segmentation, neighborhood sampling, ridge
regression on perturbed predictions). We adapt our model's torch tensor
inputs ↔ LIME's HWC numpy ↔ uint8 conventions.

Optional dependency: ``lime`` and ``scikit-image`` are listed in
``requirements.txt`` but not always installed (the bee data pipeline runs
fine without them). Importing ``lime_explanation`` only fails if you call
it without those packages installed::

    pip install lime scikit-image

For an out-of-the-box pixel-attribution alternative that needs no extra
deps, see ``occlusion.occlusion_map`` — same input/output contract,
roughly the same interpretation, slower than Grad-CAM but model-agnostic.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("explainability.lime")


def _torch_classify_fn(model: nn.Module, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that maps ``(N, H, W, 3) uint8`` → ``(N, num_classes)`` probs.

    LIME calls the classifier with arrays of perturbed images; we batch
    them through the torch model on the right device.
    """
    def _fn(arr: np.ndarray) -> np.ndarray:
        # arr: (N, H, W, 3), uint8 in [0, 255] OR float in [0, 1].
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy()
        return probs

    return _fn


def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) float tensor in [0, 1] to (H, W, 3) uint8."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must be (3, H, W); got {tuple(image.shape)}")
    arr = image.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def lime_explanation(
    model: nn.Module,
    image: torch.Tensor,
    *,
    target_class: int | None = None,
    num_samples: int = 1000,
    num_features: int = 5,
    random_seed: int = 1337,
) -> np.ndarray:
    """Run LIME on a single image and return a (H, W) attribution heatmap.

    Heatmap values are signed: positive where pixels supported the
    prediction, negative where they pushed against it. Magnitudes are
    LIME's local-linear coefficients (not normalized).

    Requires ``lime`` and ``scikit-image``::

        pip install lime scikit-image

    Raises ``ImportError`` with that hint if either is missing.
    """
    try:
        from lime import lime_image
    except ImportError as e:  # pragma: no cover - environmental
        raise ImportError(
            "lime_explanation needs the optional 'lime' and 'scikit-image' "
            "packages. Install with: pip install lime scikit-image"
        ) from e

    device = next(model.parameters()).device
    model.eval()

    img_uint8 = _to_hwc_uint8(image)
    classify = _torch_classify_fn(model, device)

    explainer = lime_image.LimeImageExplainer(random_state=random_seed)
    explanation = explainer.explain_instance(
        image=img_uint8,
        classifier_fn=classify,
        top_labels=5,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    if target_class is None:
        target_class = int(explanation.top_labels[0])

    # Map per-segment coefficients onto a per-pixel heatmap.
    segments = explanation.segments  # (H, W) int superpixel labels
    seg_coefs = dict(explanation.local_exp.get(target_class, []))
    heat = np.zeros_like(segments, dtype=np.float32)
    for seg_id, weight in seg_coefs.items():
        heat[segments == seg_id] = weight
    return heat
