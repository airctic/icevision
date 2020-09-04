__all__ = ["LightningModelAdapter"]

import pytorch_lightning as pl
from icevision.imports import *
from icevision.metrics import *


class LightningModelAdapter(pl.LightningModule, ABC):
    def __init__(self, metrics: List[Metric] = None):
        super().__init__()
        self.metrics = metrics or []

    def accumulate_metrics(self, records, preds):
        for metric in self.metrics:
            metric.accumulate(records=records, preds=preds)

    def finalize_metrics(self) -> dict:
        all_logs = {}
        for metric in self.metrics:
            metric_logs = metric.finalize()
            metric_logs = {f"{metric.name}/{k}": v for k, v in metric_logs.items()}
            all_logs.update(metric_logs)
        return all_logs
