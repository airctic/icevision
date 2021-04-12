__all__ = ["ModelAdapter"]

from icevision.models.mmdet.lightning.model_adapter import MMDetModelAdapter
from icevision.models.mmdet.common.mask import convert_raw_predictions


class ModelAdapter(MMDetModelAdapter):
    def convert_raw_predictions(self, batch, raw_preds, records):
        return convert_raw_predictions(
            batch=batch, raw_preds=raw_preds, records=records, detection_threshold=0.0
        )
