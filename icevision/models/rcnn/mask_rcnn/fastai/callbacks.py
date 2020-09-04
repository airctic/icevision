__all__ = ["MaskRCNNCallback"]

from icevision.models.rcnn.fastai.callbacks import *
from icevision.models.rcnn.mask_rcnn.prediction import *


class MaskRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(
            raw_preds=raw_preds, detection_threshold=0.0, mask_threshold=0.0
        )
