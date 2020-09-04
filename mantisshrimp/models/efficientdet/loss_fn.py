__all__ = ["loss_fn"]

from icevision.imports import *


def loss_fn(preds, targets) -> torch.Tensor:
    return preds["loss"]
