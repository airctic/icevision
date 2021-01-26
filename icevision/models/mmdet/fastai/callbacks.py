__all__ = ["MMDetectionCallback"]

from icevision.imports import *
from icevision.engines.fastai import *


class _ModelWrap(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, xb):
        return self.model.train_step(data=xb, optimizer=None)


class MMDetectionCallback(fastai.Callback):
    def after_create(self):
        self.learn.model = _ModelWrap(self.model)

    def before_batch(self):
        self.learn.records = self.yb[0]
        self.learn.yb = self.xb
