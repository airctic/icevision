__all__ = ["COCOMetric"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.models import *
from mantisshrimp.data import *
from .coco_eval import CocoEvaluator
from ..metric import *


class COCOMetric(Metric):
    def __init__(self, records, bbox=True, mask=True, keypoint=False):
        super().__init__()
        self._coco_ds = coco_api_from_records(records)

        self.iou_types = []
        if bbox:
            self.iou_types.append("bbox")
        if mask:
            self.iou_types.append("segm")
        if keypoint:
            self.iou_types.append("keypoints")

    def reset(self):
        print("##### RESET CALLED #####")
        self.coco_evaluator = CocoEvaluator(self._coco_ds, self.iou_types)

    def accumulate(self, xb, yb, preds):
        print("##### ACCUMULATE #####")
        # TODO: Implement batch_to_cpu helper function
        preds = [{k: v.to(torch.device("cpu")) for k, v in p.items()} for p in preds]
        res = {y["image_id"].item(): pred for y, pred in zip(yb, preds)}
        self.coco_evaluator.update(res)

    def finalize(self) -> float:
        print("##### FINALIZE #####")
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        return 0
