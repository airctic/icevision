__all__ = ["RetinanetCallback"]

from icevision.models.torchvision_models.fastai.callbacks import *
from icevision.models.torchvision_models.retinanet.prediction import *


class RetinanetCallback(RCNNCallback):
    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(raw_preds=raw_preds, detection_threshold=0.0)
