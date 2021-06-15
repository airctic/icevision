"""
Largely copied over from `icevision.models.ultralytics.yolov5.prectiction`, but with
classification added
"""

from icevision.utils.utils import unroll_dict
from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl

# from icevision.models.ultralytics.yolov5.dataloaders import *
from icevision.models.ultralytics.yolov5.prediction import (
    convert_raw_predictions as convert_raw_detection_predictions,
)
from icevision.models.multitask.utils.prediction import *


def convert_raw_predictions(
    batch,
    raw_detection_preds: Tensor,
    raw_classification_preds: TensorDict,
    records: Sequence[BaseRecord],
    classification_configs: dict,
    detection_threshold: float = 0.4,
    nms_iou_threshold: float = 0.6,
    keep_images: bool = False,
):
    preds = convert_raw_detection_predictions(
        batch=batch,
        raw_preds=raw_detection_preds,
        records=records,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
        keep_images=keep_images,
    )
    for pred, raw_classification_pred in zipsafe(
        preds, unroll_dict(raw_classification_preds)
    ):
        add_classification_components_to_pred_record(
            pred_record=pred.pred,
            classification_configs=classification_configs,
        )
        postprocess_and_add_classification_preds_to_record(
            gt_record=pred.ground_truth,
            pred_record=pred.pred,
            classification_configs=classification_configs,
            raw_classification_pred=raw_classification_pred,
        )

    return preds
