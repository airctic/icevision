__all__ = ["loss"]

from mantisshrimp.imports import *


def loss(preds, targets) -> torch.Tensor:
    return preds["loss"]
