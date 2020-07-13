__all__ = ["loss_fn"]

from mantisshrimp.imports import *


def loss_fn(preds, targets) -> Tensor:
    return sum(preds.values())
