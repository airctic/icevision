__all__ = ["FasterRCNNCallback"]

from icevision.models.torchvision_models.fastai.callbacks import *
from icevision.models.torchvision_models.faster_rcnn.prediction import *


class FasterRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(raw_preds=raw_preds, detection_threshold=0.0)
