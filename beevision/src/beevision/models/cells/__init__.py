"""7-class cell classifier on bee-comb cell crops.

Trained on ``deepbee_cls`` (DS-COMB-PT): 63,640 cell-level labels across
1,202 BEE_HOPE frames, mapped to 7 cell types — ``egg, larva, capped_brood,
pollen, nectar, honey, other``. ResNet-50 backbone with ImageNet pretraining,
fine-tuned end-to-end. Used as the second stage of the segmentation pipeline:
binary comb mask → cell extraction → per-cell 7-way classification.
"""
from beevision.models.cells.losses import CrossEntropyCells
from beevision.models.cells.metrics import CellMetricAccumulator
from beevision.models.cells.resnet50 import (
    CellModelConfig,
    ResNet50Cells,
    build_model,
)

__all__ = [
    "CellModelConfig",
    "ResNet50Cells",
    "build_model",
    "CrossEntropyCells",
    "CellMetricAccumulator",
]
