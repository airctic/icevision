__all__ = ["KeypointRCNNCallback"]

from icevision.models.torchvision_models.fastai.callbacks import *
from icevision.models.torchvision_models.keypoint_rcnn.prediction import *


class KeypointRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(raw_preds=raw_preds, detection_threshold=0.0)
