__all__ = ["RCNNModelAdapter"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.torchvision.loss_fn import loss_fn
from icevision.core.record_components import (
    InstanceMasksRecordComponent,
    BBoxesRecordComponent,
)
from icevision.core.mask import MaskArray
from icevision.core.bbox import BBox


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
        self.__val_predict(batch, loss_log_key="val_loss")

    def __val_predict(self, batch, loss_log_key):
        (xb, yb), records = batch

        self.train()
        train_preds = self(xb, yb)
        loss = loss_fn(train_preds, yb)

        self.eval()
        raw_preds = self(xb)
        preds = self.convert_raw_predictions(
            batch=batch, raw_preds=raw_preds, records=records
        )
        self.accumulate_metrics(preds=preds)

        self.log(loss_log_key, loss)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()

    def test_step(self, batch, batch_idx):
        self.__val_predict(batch=batch, loss_log_key="test_loss")

    def test_epoch_end(self, outs):
        self.finalize_metrics()
