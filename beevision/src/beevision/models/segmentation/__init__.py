"""Frame-level binary (comb vs background) segmentation on ``deepbee_seg``.

The pipeline diagram says 7 classes but that's a two-stage design: this model
produces a comb/background mask, and ``deepbee_cls`` handles per-cell
classification into the 7 cell classes separately (DeepBee paper layout).
"""
from beevision.models.segmentation.losses import CEDiceLoss, SoftDiceLoss
from beevision.models.segmentation.metrics import SegMetricAccumulator
from beevision.models.segmentation.unet_dinov2 import (
    DINOv2Encoder,
    DINOv2UNet,
    SegModelConfig,
    UNetDecoder,
    build_model,
)

__all__ = [
    "DINOv2Encoder",
    "DINOv2UNet",
    "SegModelConfig",
    "UNetDecoder",
    "build_model",
    "CEDiceLoss",
    "SoftDiceLoss",
    "SegMetricAccumulator",
]
