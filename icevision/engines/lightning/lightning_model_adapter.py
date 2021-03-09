__all__ = ["LightningModelAdapter"]

import pytorch_lightning as pl
from icevision.imports import *
from icevision.metrics import *


class LightningModelAdapter(pl.LightningModule, ABC):
    def __init__(self, metrics: List[Metric] = None):
        super().__init__()
        self.metrics = metrics or []

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def finalize_metrics(self) -> None:
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                self.log(f"{metric.name}/{k}", v)
