__all__ = ["loss_fn"]

from icevision.imports import *


def loss_fn(preds, targets) -> Tensor:
    return sum(preds.values())
