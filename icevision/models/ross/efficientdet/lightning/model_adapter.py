__all__ = ["ModelAdapter"]

from icevision.imports import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.ross import efficientdet


class ModelAdapter(LightningModelAdapter, ABC):
    """Lightning module specialized for EfficientDet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), records = batch
        preds = self(xb, yb)

        loss = self.compute_loss(preds, yb)

        for k, v in preds.items():
            self.log(f"train_{k}", v)

        return loss

    def compute_loss(self, preds, yb):
        return efficientdet.loss_fn(preds, yb)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, loss_log_key="val")

    def _shared_eval(self, batch, loss_log_key):
        (xb, yb), records = batch

        raw_preds = self(xb, yb)

        preds = self.convert_raw_predictions(xb, yb, raw_preds, records)

        self.compute_loss(raw_preds, yb)

        self.accumulate_metrics(preds)

        for k, v in raw_preds.items():
            if "loss" in k:
                self.log(f"{loss_log_key}_{k}", v)

    def convert_raw_predictions(self, xb, yb, raw_preds, records):
        # Note: raw_preds["detections"] key is available only during Pytorch Lightning validation/test step
        # Calling the method manually (instead of letting the Trainer call it) will raise an exception.
        return efficientdet.convert_raw_predictions(
            batch=(xb, yb),
            raw_preds=raw_preds["detections"],
            records=records,
            detection_threshold=0.0,
        )

    def validation_epoch_end(self, outs):
        self.finalize_metrics()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch=batch, loss_log_key="test")

    def test_epoch_end(self, outs):
        self.finalize_metrics()
