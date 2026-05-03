"""Tests for the explainability module.

We use a tiny conv classifier for Grad-CAM and occlusion tests so each test
runs in milliseconds. The fake model exposes the same ``.backbone.layer4``
structure as ``ResNet50Mite``/``ResNet50Cells`` so ``find_default_target_layer``
can find it.

LIME tests are skipped when ``lime``/``skimage`` aren't installed (they
ship in ``requirements.txt`` but aren't always present).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
import torch.nn as nn

from beevision.explainability import (
    GradCAM,
    OcclusionConfig,
    apply_colormap,
    apply_signed_colormap,
    find_default_target_layer,
    gradcam,
    occlusion,
    occlusion_map,
    overlay_heatmap,
)


# ---------- Tiny model with ResNet-style attributes -----------------------


class _TinyResNetLike(nn.Module):
    """Mimics ``ResNet50Mite``/``ResNet50Cells`` enough for the explainers.

    Exposes ``.backbone.layer4`` (a tiny conv block) and a final FC head, so
    Grad-CAM's ``find_default_target_layer`` resolves correctly.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # Build a minimal "backbone" namespace.
        backbone = nn.Module()
        backbone.stem = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        # A tiny "layer4" with one block — Grad-CAM hooks into the last child.
        backbone.layer4 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
        )
        backbone.gap = nn.AdaptiveAvgPool2d(1)
        backbone.fc = nn.Linear(16, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.stem(x)
        x = self.backbone.layer4(x)
        x = self.backbone.gap(x).flatten(1)
        return self.backbone.fc(x)


def _make_model(num_classes: int = 2) -> _TinyResNetLike:
    torch.manual_seed(0)
    m = _TinyResNetLike(num_classes=num_classes)
    m.eval()
    return m


# ---------- Grad-CAM ------------------------------------------------------


def test_gradcam_default_target_layer_resolves() -> None:
    model = _make_model()
    layer = find_default_target_layer(model)
    # Should be the last child of layer4.
    assert layer is model.backbone.layer4[-1]


def test_gradcam_default_target_layer_errors_without_backbone() -> None:
    class _NoBackbone(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x
    with pytest.raises(AttributeError, match="backbone"):
        find_default_target_layer(_NoBackbone())


def test_gradcam_returns_correct_shape_and_range() -> None:
    model = _make_model(num_classes=3)
    x = torch.randn(2, 3, 56, 56)
    cam = gradcam(model, x, target_class=1)
    assert cam.shape == (2, 56, 56)
    assert cam.dtype == np.float32
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_gradcam_single_image_input_drops_batch_dim() -> None:
    model = _make_model(num_classes=2)
    x = torch.randn(3, 56, 56)
    cam = gradcam(model, x)
    assert cam.shape == (56, 56)


def test_gradcam_target_class_list_per_sample() -> None:
    model = _make_model(num_classes=3)
    x = torch.randn(2, 3, 56, 56)
    cam = gradcam(model, x, target_class=[0, 2])
    assert cam.shape == (2, 56, 56)


def test_gradcam_target_class_list_length_mismatch_raises() -> None:
    model = _make_model(num_classes=3)
    x = torch.randn(2, 3, 56, 56)
    with pytest.raises(ValueError, match="length"):
        gradcam(model, x, target_class=[0, 1, 2])


def test_gradcam_default_uses_argmax() -> None:
    """No target_class → use the model's prediction; should match explicit pass-through."""
    model = _make_model(num_classes=3)
    x = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        argmax = int(model(x).argmax(dim=1).item())
    cam_default = gradcam(model, x)
    cam_explicit = gradcam(model, x, target_class=argmax)
    np.testing.assert_allclose(cam_default, cam_explicit, atol=1e-5)


def test_gradcam_must_be_used_in_context_manager() -> None:
    model = _make_model()
    cam = GradCAM(model, find_default_target_layer(model))
    with pytest.raises(RuntimeError, match="with"):
        cam(torch.randn(1, 3, 56, 56))


def test_gradcam_localizes_bright_region() -> None:
    """A model trained to pick "right half is class 1" should put CAM mass on the right.

    We train the tiny model briefly on a left/right brightness signal, then
    check that the Grad-CAM heatmap for class 1 has higher mean activation
    on the right half than the left.
    """
    torch.manual_seed(0)
    model = _make_model(num_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()

    B, S = 8, 56
    x = torch.zeros(B, 3, S, S)
    x[B // 2:, :, :, S // 2:] = 1.0  # second half: bright on right
    y = torch.tensor([0] * (B // 2) + [1] * (B // 2))
    for _ in range(60):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()

    model.eval()
    test_img = torch.zeros(1, 3, S, S)
    test_img[:, :, :, S // 2:] = 1.0  # right-bright; should predict class 1
    cam = gradcam(model, test_img, target_class=1)
    left_mean = cam[:, :S // 2].mean()
    right_mean = cam[:, S // 2:].mean()
    assert right_mean > left_mean, f"CAM not localized: left={left_mean:.3f} right={right_mean:.3f}"


# ---------- Occlusion -----------------------------------------------------


def test_occlusion_returns_correct_shape_and_range() -> None:
    model = _make_model(num_classes=3)
    x = torch.randn(2, 3, 56, 56)
    cfg = OcclusionConfig(patch_size=14, stride=14)
    heat = occlusion_map(model, x, target_class=1, cfg=cfg)
    assert heat.shape == (2, 56, 56)
    assert heat.dtype == np.float32
    assert heat.min() >= 0.0 and heat.max() <= 1.0


def test_occlusion_single_image_drops_batch_dim() -> None:
    model = _make_model(num_classes=2)
    x = torch.randn(3, 56, 56)
    cfg = OcclusionConfig(patch_size=14, stride=14)
    heat = occlusion(model, x, cfg=cfg)
    assert heat.shape == (56, 56)


class _LeftRightLinear(nn.Module):
    """Hand-crafted binary classifier: logits = [mean(left_half), mean(right_half)] * scale.

    Deterministic — no training. Class 1 is "right half is bright"; occluding
    the right half of a right-bright image directly tanks the class-1 logit
    while occluding the left half has no effect.
    """

    def __init__(self, scale: float = 20.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = x.shape[-1]
        left = x[..., :, :W // 2].mean(dim=(1, 2, 3))
        right = x[..., :, W // 2:].mean(dim=(1, 2, 3))
        return torch.stack([left, right], dim=1) * self.scale


def test_occlusion_localizes_bright_region_deterministic() -> None:
    """Hand-built left/right model: occluding right half of right-bright must drop class 1."""
    model = _LeftRightLinear()
    model.eval()
    S = 56
    img = torch.zeros(1, 3, S, S)
    img[:, :, :, S // 2:] = 1.0  # right-bright

    cfg = OcclusionConfig(patch_size=14, stride=14)
    heat = occlusion_map(model, img, target_class=1, cfg=cfg)
    left_mean = float(heat[0, :, :S // 2].mean())
    right_mean = float(heat[0, :, S // 2:].mean())
    assert right_mean > left_mean, (
        f"occlusion not localized: left={left_mean:.3f} right={right_mean:.3f}"
    )


def test_occlusion_target_class_length_mismatch_raises() -> None:
    model = _make_model(num_classes=3)
    x = torch.randn(2, 3, 56, 56)
    with pytest.raises(ValueError, match="length"):
        occlusion_map(model, x, target_class=[0, 1, 2])


# ---------- Visualize -----------------------------------------------------


def test_apply_colormap_jet_shape_and_range() -> None:
    h = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16).astype(np.float32)
    rgb = apply_colormap(h, cmap="jet")
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8
    # Endpoints: 0 → blue dominant; 1 → red dominant in jet.
    assert rgb[0, 0, 2] >= rgb[0, 0, 0]   # blue ≥ red at low end
    assert rgb[-1, -1, 0] >= rgb[-1, -1, 2]  # red ≥ blue at high end


def test_apply_colormap_inferno_runs() -> None:
    h = np.zeros((4, 4), dtype=np.float32)
    rgb = apply_colormap(h, cmap="inferno")
    assert rgb.shape == (4, 4, 3)


def test_apply_colormap_unknown_cmap_raises() -> None:
    with pytest.raises(ValueError, match="unknown colormap"):
        apply_colormap(np.zeros((4, 4)), cmap="rainbow_xyz")


def test_apply_colormap_clips_out_of_range() -> None:
    h = np.array([[-1.0, 0.5], [0.0, 2.0]], dtype=np.float32)
    rgb = apply_colormap(h)  # should not raise; clipped internally
    assert rgb.shape == (2, 2, 3)


def test_apply_signed_colormap_handles_zero_input() -> None:
    rgb = apply_signed_colormap(np.zeros((4, 4)))
    assert rgb.shape == (4, 4, 3)


def test_apply_signed_colormap_normalizes_by_max_abs() -> None:
    h = np.array([[-2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    rgb = apply_signed_colormap(h, cmap="jet")
    # The pixels at |2.0| (positions [0,0] and [1,1]) should be the same color,
    # since signed colormap uses |x| / max|x|.
    assert np.array_equal(rgb[0, 0], rgb[1, 1])


def test_overlay_heatmap_returns_uint8_with_image_shape() -> None:
    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    h = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16).astype(np.float32)
    out = overlay_heatmap(img, h, alpha=0.5)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_overlay_heatmap_resizes_when_shapes_differ() -> None:
    img = np.full((20, 24, 3), 50, dtype=np.uint8)
    h = np.zeros((10, 12), dtype=np.float32)
    out = overlay_heatmap(img, h)
    assert out.shape == (20, 24, 3)


def test_overlay_heatmap_alpha_zero_is_image_only() -> None:
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    h = np.ones((8, 8), dtype=np.float32)
    out = overlay_heatmap(img, h, alpha=0.0)
    np.testing.assert_array_equal(out, img)


def test_overlay_heatmap_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        overlay_heatmap(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4)), alpha=1.5)


# ---------- LIME (skipped if dep missing) --------------------------------


_LIME_AVAILABLE = (
    importlib.util.find_spec("lime") is not None
    and importlib.util.find_spec("skimage") is not None
)


@pytest.mark.skipif(not _LIME_AVAILABLE, reason="lime/scikit-image not installed")
def test_lime_explanation_returns_pixel_heatmap() -> None:
    from beevision.explainability import lime_explanation

    model = _make_model(num_classes=2)
    img = torch.rand(3, 56, 56)
    heat = lime_explanation(model, img, target_class=0, num_samples=50)
    assert heat.shape == (56, 56)


@pytest.mark.skipif(_LIME_AVAILABLE, reason="lime present; ImportError test n/a")
def test_lime_explanation_raises_clean_error_when_dep_missing() -> None:
    from beevision.explainability import lime_explanation

    model = _make_model()
    img = torch.rand(3, 56, 56)
    with pytest.raises(ImportError, match="pip install lime"):
        lime_explanation(model, img)
