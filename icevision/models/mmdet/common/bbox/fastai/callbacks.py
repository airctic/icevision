__all__ = ["BBoxMMDetectionCallback"]

from icevision.models.mmdet.fastai.callbacks import MMDetectionCallback
from icevision.models.mmdet.common.bbox.prediction import convert_raw_predictions


class BBoxMMDetectionCallback(MMDetectionCallback):
    def convert_raw_predictions(self, batch, raw_preds, records):
        return convert_raw_predictions(
            batch=batch, raw_preds=raw_preds, records=records, detection_threshold=0.0
        )
