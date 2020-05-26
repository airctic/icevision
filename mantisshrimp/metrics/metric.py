__all__ = ["Metric"]

from ..utils import *


class Metric:
    def __init__(self):
        self._model = None

    def step(self, xb, yb, preds):
        raise NotImplementedError

    def end(self, outs):
        raise NotImplementedError

    def register_model(self, model):
        self._model = model

    @property
    def model(self):
        if notnone(self._model):
            return self._model
        raise RuntimeError(
            "Register a model with `register_model` before using the metric"
        )
