__all__ = ["FastaiMetricAdapter"]

from mantisshrimp.imports import *
from mantisshrimp.metrics import Metric
from mantisshrimp.engines.fastai.imports import *


class FastaiMetricAdapter(fastai.Metric):
    def __init__(self, metric: Metric):
        self.metric = metric

    def reset(self):
        pass

    def accumulate(self, learn: fastai.Learner):
        self.metric.accumulate(records=learn.records, preds=learn.converted_preds)

    @property
    def value(self) -> Dict[str, float]:
        return self.metric.finalize()

    @property
    def name(self) -> str:
        return self.metric.name
