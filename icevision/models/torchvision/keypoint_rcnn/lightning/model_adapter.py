__all__ = ["ModelAdapter"]

from icevision.models.torchvision.lightning.model_adapter import *
from icevision.models.torchvision.keypoint_rcnn.prediction import *


class ModelAdapter(RCNNModelAdapter):
    """Lightning module specialized for keypoint_rcnn, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def convert_raw_predictions(self, batch, raw_preds, records):
        return convert_raw_predictions(
            batch=batch, raw_preds=raw_preds, records=records, detection_threshold=0.0
        )
