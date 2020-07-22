__all__ = ["FasterRCNNCallback"]

from mantisshrimp.models.rcnn.fastai.callbacks import *
from mantisshrimp.models.rcnn.faster_rcnn.prediction import *


class FasterRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(raw_preds=raw_preds, detection_threshold=0.0)
