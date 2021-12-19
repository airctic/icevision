__all__ = ["FasterRCNNCallback"]

from icevision.models.torchvision.fastai.callbacks import *
from icevision.models.torchvision.faster_rcnn.prediction import *


class FasterRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, batch, raw_preds, detection_threshold=0.0):
        return convert_raw_predictions(
            batch=batch,
            raw_preds=raw_preds,
            records=self.learn.records,
            detection_threshold=detection_threshold,
        )
