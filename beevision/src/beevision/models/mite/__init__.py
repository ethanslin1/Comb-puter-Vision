"""Binary mite classifier on bee crops.

Trained on combined ``varroa`` (Zenodo 4085044) + ``beeimage`` (Kaggle)
sources; ~11k train / ~2.5k val / ~4k test bee-crop images, label
``{0=no_mite, 1=mite}``. ResNet-50 backbone with ImageNet pretraining,
fine-tuned end-to-end (no encoder freeze by default).
"""
from beevision.models.mite.losses import CrossEntropyMite
from beevision.models.mite.metrics import ClassificationMetricAccumulator
from beevision.models.mite.resnet50 import (
    MiteModelConfig,
    ResNet50Mite,
    build_model,
)

__all__ = [
    "MiteModelConfig",
    "ResNet50Mite",
    "build_model",
    "CrossEntropyMite",
    "ClassificationMetricAccumulator",
]
