__all__ = ["MaskMMSegmentationCallback"]

from icevision.models.mmseg.fastai.callbacks import MMSegmentationCallback
from icevision.models.mmseg.common.mask.prediction import convert_raw_predictions


class MaskMMSegmentationCallback(MMSegmentationCallback):
    def convert_raw_predictions(self, batch, raw_preds, records):
        return convert_raw_predictions(
            batch=batch, raw_preds=raw_preds, records=records
        )
