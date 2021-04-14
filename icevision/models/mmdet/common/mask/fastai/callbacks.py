__all__ = ["MaskMMDetectionCallback"]

from icevision.models.mmdet.fastai.callbacks import MMDetectionCallback
from icevision.models.mmdet.common.mask.prediction import convert_raw_predictions


class MaskMMDetectionCallback(MMDetectionCallback):
    def convert_raw_predictions(self, batch, raw_preds, records):
        return convert_raw_predictions(
            batch=batch, raw_preds=raw_preds, records=records, detection_threshold=0.0
        )
