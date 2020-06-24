__all__ = ["COCOMetric"]

from mantisshrimp.imports import *
from mantisshrimp.data import *
from mantisshrimp.metrics.coco_metric.coco_eval import CocoEvaluator
from mantisshrimp.metrics.metric import *


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

        self.reset()

    def reset(self):
        self._coco_evaluator = CocoEvaluator(self._coco_ds, self.iou_types)

    def accumulate(self, xb, yb, preds):
        # TODO: Implement batch_to_cpu helper function
        assert len(xb) == len(yb) == len(preds)
        preds = [{k: v.to(torch.device("cpu")) for k, v in p.items()} for p in preds]
        res = {y["image_id"].item(): pred for y, pred in zip(yb, preds)}
        self._coco_evaluator.update(res)

    def finalize(self) -> Dict[str, float]:
        self._coco_evaluator.synchronize_between_processes()
        self._coco_evaluator.accumulate()
        self._coco_evaluator.summarize()
        logs = {
            f"{k}_ap": float(v.stats[0])
            for k, v in self._coco_evaluator.coco_eval.items()
        }
        self.reset()
        return logs
