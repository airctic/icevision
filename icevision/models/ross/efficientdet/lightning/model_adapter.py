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

        loss = efficientdet.loss_fn(preds, yb)

        for k, v in preds.items():
            self.log(f"train/{k}", v)

        return loss

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch

        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions(
                batch=(xb, yb),
                raw_preds=raw_preds["detections"],
                records=records,
                detection_threshold=0.0,
            )
            loss = efficientdet.loss_fn(raw_preds, yb)

        self.accumulate_metrics(preds)

        for k, v in raw_preds.items():
            if "loss" in k:
                self.log(f"valid/{k}", v)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
