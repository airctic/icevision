__all__ = ["RetinanetCallback"]

from icevision.models.torchvision.fastai.callbacks import *
from icevision.models.torchvision.retinanet.prediction import *


class RetinanetCallback(RCNNCallback):
    def convert_raw_predictions(self, batch, raw_preds, detection_threshold=0.0):
        return convert_raw_predictions(
            batch=batch,
            raw_preds=raw_preds,
            records=self.learn.records,
            detection_threshold=detection_threshold,
        )
