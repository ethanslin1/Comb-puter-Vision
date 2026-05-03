"""Explainability tools for the BeeVision classifiers.

Three attribution methods, all targeting the same single-image output:
a per-pixel heatmap that says "where did the model look."

- ``gradcam`` — gradient-weighted class activation maps (Selvaraju 2017).
  Pure-torch, fast, requires picking a target conv layer (we provide a
  default for ResNet-50). Best signal-to-noise for our use case.
- ``occlusion`` — sliding-window occlusion sensitivity (Zeiler 2014).
  Pure-torch, model-agnostic, no layer choice, slower than CAM.
- ``lime`` — superpixel + ridge-regression attribution (Ribeiro 2016).
  Wraps the third-party ``lime`` package; raises a clean ImportError if
  ``lime`` and ``scikit-image`` aren't installed. Use ``occlusion`` if
  you want a no-deps cousin.

Plus visualization helpers (``apply_colormap``, ``overlay_heatmap``)
for blending heatmaps onto frames for the dashboard.
"""
from beevision.explainability.gradcam import (
    GradCAM,
    find_default_target_layer,
    gradcam,
)
from beevision.explainability.lime import lime_explanation
from beevision.explainability.occlusion import (
    OcclusionConfig,
    occlusion,
    occlusion_map,
)
from beevision.explainability.visualize import (
    apply_colormap,
    apply_signed_colormap,
    overlay_heatmap,
)

__all__ = [
    # Grad-CAM
    "GradCAM",
    "gradcam",
    "find_default_target_layer",
    # Occlusion
    "OcclusionConfig",
    "occlusion",
    "occlusion_map",
    # LIME
    "lime_explanation",
    # Visualization
    "apply_colormap",
    "apply_signed_colormap",
    "overlay_heatmap",
]
