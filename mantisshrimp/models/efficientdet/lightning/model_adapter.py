__all__ = ["ModelAdapter"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.metrics import *
from mantisshrimp.engines.lightning.lightning_model_adapter import LightningModelAdapter
from mantisshrimp.models.rcnn.loss_fn import loss_fn


class ModelAdapter(LightningModelAdapter, ABC):
    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        preds = self(xb, yb)
        loss = loss_fn(preds, yb)
        log = {f"train/{k}": v for k, v in preds.items()}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        xb, yb = batch

        with torch.no_grad():
            preds = self(xb, yb)
            loss = loss_fn(preds, yb)

        self.accumulate_metrics(xb, yb, preds)

        log = {f"valid/{k}": v for k, v in preds.items() if "loss" in k}
        return {"valid/loss": loss, "log": log}

    def validation_epoch_end(self, outs):
        merged_logs = mergeds(outs)
        set_trace()
        avg_loss = torch.stack(merged_logs["valid/loss"]).mean()
        avg_logs = {k: torch.stack(v).mean() for k, v in merged_logs["log"].items()}

        metrics_log = self.finalize_metrics()

        log = {**avg_logs, **metrics_log}
        return {"val_loss": avg_loss, "log": log}
