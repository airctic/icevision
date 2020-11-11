__all__ = ["RCNNModelAdapter"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.torchvision_models.loss_fn import loss_fn


class RCNNModelAdapter(LightningModelAdapter, ABC):
    def __init__(self, model: nn.Module, metrics: Sequence[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    @abstractmethod
    def convert_raw_predictions(self, raw_preds):
        """Convert raw predictions from the model to library standard."""

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), records = batch
        preds = self(xb, yb)

        loss = loss_fn(preds, yb)
        log = {"train/loss": loss}

        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch

        with torch.no_grad():
            self.train()
            train_preds = self(xb, yb)
            loss = loss_fn(train_preds, yb)

            self.eval()
            raw_preds = self(xb)
            preds = self.convert_raw_predictions(raw_preds=raw_preds)
            self.accumulate_metrics(records=records, preds=preds)

        return {"valid/loss": loss}

    def validation_epoch_end(self, outs):
        loss_log = {k: torch.stack(v).mean() for k, v in mergeds(outs).items()}
        metrics_log = self.finalize_metrics()
        log = {**loss_log, **metrics_log}
        return {"val_loss": log["valid/loss"], "log": log}
