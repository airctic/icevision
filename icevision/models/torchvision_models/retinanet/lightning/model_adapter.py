__all__ = ["ModelAdapter"]

from icevision.models.torchvision_models.lightning.model_adapter import *
from icevision.models.torchvision_models.retinanet.prediction import *


class ModelAdapter(RCNNModelAdapter):
    """Lightning module specialized for retinanet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(raw_preds=raw_preds, detection_threshold=0.0)
