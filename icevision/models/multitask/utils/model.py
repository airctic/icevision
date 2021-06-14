from enum import Enum

__all__ = ["ForwardType"]


class ForwardType(Enum):
    TRAIN_MULTI_AUG = 1
    TRAIN = 2
    EVAL = 3
