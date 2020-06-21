__all__ = ["FastaiMetricAdapter"]

from mantisshrimp.metrics import Metric
from mantisshrimp.engines.fastai.imports import *
from fastai2.learner import Learner


class FastaiMetricAdapter(fastai.Metric):
    def __init__(self, metric: Metric):
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def accumulate(self, learn: fastai.Learner):
        self.metric.accumulate(*learn.xb, *learn.yb, learn.pred)

    @property
    def value(self):
        return self.metric.finalize()
