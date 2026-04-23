"""Albumentations train/eval pipelines for BeeVision.

The spec (#9) nails down the train pipeline:

    HFlip, Rotate±15, RandomBrightnessContrast, HueSaturationValue,
    CoarseDropout, GaussianBlur, JPEG(40-90)

Plus: mirror geometry to masks for segmentation (so anything that moves
pixels must use ``additional_targets`` or albumentations' built-in mask
handling), and eval = resize + reflect-pad only. WeightedRandomSampler on
the train mite set lives in ``datasets.py``; that's data-side, not
augmentation-side.

We expose three builders:

- ``classification_transforms(image_size, train)`` → image-only transforms
  for cell classification (deepbee_cls) and mite classification
  (varroa, beeimage). Train is the full pipeline; eval is resize+pad.
- ``segmentation_transforms(image_size, train)`` → image + mask
  transforms for the UNet; same train pipeline minus color ops applied
  to masks (albumentations handles this automatically; we just declare
  the targets).
- ``apply(tf, image, mask=None)`` → thin convenience wrapper that
  returns a dict with numpy arrays (float32 image in CHW, int64 mask).

Determinism: albumentations pipelines are seeded on first use of the RNG
per-thread. For reproducible tests we expose ``seed_pipeline(tf, seed)``.
"""
from __future__ import annotations

import random
from typing import Any, Mapping

import albumentations as A
import numpy as np

__all__ = [
    "classification_transforms",
    "segmentation_transforms",
    "apply",
    "seed_pipeline",
]


# ---------- Builders -------------------------------------------------------


def _train_color_block() -> list[A.BasicTransform]:
    """Color / photometric ops: applied to images only, not masks (albu default)."""
    return [
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ImageCompression(quality_range=(40, 90), p=0.3),
    ]


def _train_geom_block(image_size: int) -> list[A.BasicTransform]:
    """Geometric ops: applied to image and mask together."""
    return [
        A.HorizontalFlip(p=0.5),
        A.Rotate(
            limit=15,
            border_mode=0,  # cv2.BORDER_CONSTANT
            fill=0,
            fill_mask=0,
            p=0.5,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            fill=0,
            p=0.3,
        ),
    ]


def _eval_resize_pad(image_size: int) -> list[A.BasicTransform]:
    """Eval-time: resize short edge then reflect-pad to square."""
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=2,  # cv2.BORDER_REFLECT_101
        ),
    ]


def classification_transforms(image_size: int = 224, *, train: bool) -> A.Compose:
    """Image-only pipeline for cell/mite classification.

    Train: geom + color; eval: resize+reflect-pad only.
    """
    if train:
        ops: list[A.BasicTransform] = [
            *_eval_resize_pad(image_size),
            *_train_geom_block(image_size),
            *_train_color_block(),
        ]
    else:
        ops = _eval_resize_pad(image_size)
    return A.Compose(ops)


def segmentation_transforms(image_size: int = 512, *, train: bool) -> A.Compose:
    """Image+mask pipeline for the UNet. Geom ops mirror to the mask.

    Eval is resize-pad on both image and mask; reflection padding on the
    mask uses reflect-101 as well, since boundary rows repeat the nearest
    class index and training reduces this to "ignore".
    """
    if train:
        ops = [
            *_eval_resize_pad(image_size),
            *_train_geom_block(image_size),
            *_train_color_block(),
        ]
    else:
        ops = _eval_resize_pad(image_size)
    # additional_targets is not strictly needed when using ``mask=`` in
    # the call, but making it explicit documents the contract.
    return A.Compose(ops, additional_targets={"mask": "mask"})


# ---------- Convenience apply ---------------------------------------------


def apply(
    tf: A.Compose,
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> Mapping[str, np.ndarray]:
    """Run ``tf`` and post-normalize to model-ready arrays.

    Returns a dict with:
      - "image": float32 CHW in [0, 1]
      - "mask":  int64 HW class indices (only if ``mask`` was provided)
    """
    payload: dict[str, Any] = {"image": image}
    if mask is not None:
        payload["mask"] = mask
    out = tf(**payload)

    img = out["image"]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    # HWC → CHW
    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
        img = np.transpose(img, (2, 0, 1))

    result: dict[str, np.ndarray] = {"image": img}
    if "mask" in out:
        result["mask"] = out["mask"].astype(np.int64)
    return result


# ---------- Seeding --------------------------------------------------------


def seed_pipeline(tf: A.Compose, seed: int) -> A.Compose:
    """Deterministic pipeline runs.

    Albumentations ≥2.0 owns its own RNG per ``Compose``. We reseed it
    directly if the method is available, else fall back to seeding the
    python + numpy globals (for older 1.x installs).
    """
    setter = getattr(tf, "set_random_seed", None)
    if callable(setter):
        setter(seed)
    else:  # pragma: no cover - albumentations 1.x fallback
        random.seed(seed)
        np.random.seed(seed)
    return tf
