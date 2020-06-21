__all__ = ["FastaiMetric", "FastaiMetricAdapter"]

from mantisshrimp.metrics import Metric
from fastai2.metrics import Metric as FastaiMetric
from fastai2.learner import Learner


class FastaiMetricAdapter(FastaiMetric):
    def __init__(self, metric: Metric):
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def accumulate(self, learn: Learner):
        self.metric.accumulate(*learn.xb, *learn.yb, learn.pred)

    @property
    def value(self):
        return self.metric.finalize()
