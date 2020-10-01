__all__ = ["FastaiMetricAdapter"]

from icevision.imports import *
from icevision.metrics import Metric
from icevision.engines.fastai.imports import *


class FastaiMetricAdapter(fastai.Metric):
    def __init__(self, metric: Metric):
        self.metric = metric

    def reset(self):
        pass

    def accumulate(self, learn: fastai.Learner):
        self.metric.accumulate(records=learn.records, preds=learn.converted_preds)

    @property
    def value(self) -> Dict[str, float]:
        # return self.metric.finalize()
        # HACK: Return single item from dict
        logs = self.metric.finalize()
        return next(iter(logs.values()))

    @property
    def name(self) -> str:
        return self.metric.name
