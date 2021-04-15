__all__ = ["ModelAdapter"]

from icevision.imports import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.ultralytics import yolov5
from yolov5.utils.loss import ComputeLoss


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
        self.compute_loss = ComputeLoss(model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), _ = batch
        preds = self(xb)

        loss = self.compute_loss(preds, yb)[0]

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch

        with torch.no_grad():
            inference_out, training_out = self(xb)
            preds = yolov5.convert_raw_predictions(
                batch=xb,
                raw_preds=inference_out,
                records=records,
                detection_threshold=0.001,
                nms_iou_threshold=0.6,
            )
            loss = self.compute_loss(training_out, yb)[0]

        self.accumulate_metrics(preds)

        self.log("val_loss", loss)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
