__all__ = ["FasterRCNNCallback"]

from icevision.models.rcnn.fastai.callbacks import *
from icevision.models.rcnn.faster_rcnn.prediction import *


class FasterRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(raw_preds=raw_preds, detection_threshold=0.0)
