__all__ = ["loss"]

from mantisshrimp.imports import *


def loss(preds, targets) -> Tensor:
    return sum(preds.values())
