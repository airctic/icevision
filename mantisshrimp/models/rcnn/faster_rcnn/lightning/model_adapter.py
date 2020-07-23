__all__ = ["ModelAdapter"]

from mantisshrimp.models.rcnn.lightning.model_adapter import *
from mantisshrimp.models.rcnn.faster_rcnn.prediction import *


class ModelAdapter(RCNNModelAdapter):
    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(raw_preds=raw_preds, detection_threshold=0.0)
