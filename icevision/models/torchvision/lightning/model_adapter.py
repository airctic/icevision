__all__ = ["RCNNModelAdapter"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.torchvision.loss_fn import loss_fn


class RCNNModelAdapter(LightningModelAdapter, ABC):
    def __init__(self, model: nn.Module, metrics: Sequence[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    @abstractmethod
    def convert_raw_predictions(self, batch, raw_preds, records):
        """Convert raw predictions from the model to library standard."""

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), records = batch
        preds = self(xb, yb)

        loss = loss_fn(preds, yb)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch

        with torch.no_grad():
            self.train()
            train_preds = self(xb, yb)
            loss = loss_fn(train_preds, yb)

            self.eval()
            raw_preds = self(xb)
            preds = self.convert_raw_predictions(
                batch=batch, raw_preds=raw_preds, records=records
            )
            self.accumulate_metrics(preds=preds)

        self.log("val_loss", loss)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
