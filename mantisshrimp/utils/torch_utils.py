__all__ = ["to_np", "requires_grad", "model_device", "params"]

from mantisshrimp.imports import *


def to_np(t):
    return t.detach().cpu().numpy()


def requires_grad(model, layer):
    return list(model.parameters())[layer].requires_grad


def model_device(model):
    return first(model.parameters()).device


def params(m):
    return list(m.parameters())
