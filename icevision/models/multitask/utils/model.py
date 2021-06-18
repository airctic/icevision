from enum import Enum

__all__ = ["ForwardType"]


class ForwardType(Enum):
    TRAIN_MULTI_AUG = 1
    TRAIN = 2
    EVAL = 3
    INFERENCE = 4
    # EXPORT_ONNX = 5
    # EXPORT_TORCHSCRIPT = 6
    # EXPORT_COREML = 7
