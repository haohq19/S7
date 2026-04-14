"""S7: Selective and Simplified State Space Model.

Reference: https://arxiv.org/abs/2410.03464

Structure:
    s7.ssm        S7 SSM layer (selective Δ + diagonal complex Λ).
    s7.init       HiPPO-LegS initialization, parameter init helpers.
    s7.layers     SequenceLayer, SequenceStage, EventPooling.
    s7.model      StackedEncoder, ClassificationModel, RetrievalModel.
    s7.train.*    Optimizer setup, train/eval steps, trainer loop, checkpoints.
    s7.data.*     Dataset loaders + collate + augmentation transforms.
"""

from s7.ssm import S7, build_s7
from s7.model import ClassificationModel, BatchClassificationModel, RetrievalModel, StackedEncoder

__all__ = [
    "S7",
    "build_s7",
    "ClassificationModel",
    "BatchClassificationModel",
    "RetrievalModel",
    "StackedEncoder",
]
