__all__ = ["ModelAdapter"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.models import *
from mantisshrimp.metrics import *
from mantisshrimp.engines.lightning.lightning_model_adapter import LightningModelAdapter
from mantisshrimp.models.rcnn.loss import loss as loss_fn


class ModelAdapter(LightningModelAdapter, ABC):
    def __init__(self, model: MantisRCNN, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        preds = self(xb, yb)
        loss = loss_fn(preds, yb)
        log = {"train/loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        with torch.no_grad():
            self.train()
            preds = self(xb, yb)
            loss = loss_fn(preds, yb)
            self.eval()
            preds = self(xb)
            self.accumulate_metrics(xb, yb, preds)
        return {"valid/loss": loss}

    def validation_epoch_end(self, outs):
        loss_log = {k: torch.stack(v).mean() for k, v in mergeds(outs).items()}
        metrics_log = self.finalize_metrics()
        log = {**loss_log, **metrics_log}
        return {"val_loss": log["valid/loss"], "log": log}
