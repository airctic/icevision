__all__ = ["MaskRCNNCallback"]

from mantisshrimp.models.rcnn.fastai.callbacks import *
from mantisshrimp.models.rcnn.mask_rcnn.prediction import *


class MaskRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, raw_preds):
        return convert_raw_predictions(
            raw_preds=raw_preds, detection_threshold=0.0, mask_threshold=0.0
        )
