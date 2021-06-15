from enum import Enum

__all__ = ["ForwardType"]


class ForwardType(Enum):
    TRAIN_MULTI_AUG = 1
    TRAIN = 2
    EVAL = 3
    EXPORT_ONNX = 4
    EXPORT_TORCHSCRIPT = 5
    EXPORT_COREML = 6
