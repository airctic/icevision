__all__ = ["COCOMetric"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.models import *
from mantisshrimp.data import *
from .coco_eval import CocoEvaluator
from ..metric import *


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, MantisMaskRCNN):
        iou_types.append("segm")
    #     if isinstance(model_without_ddp, KeypointRCNN):
    #         iou_types.append("keypoints")
    return iou_types


class COCOMetric(Metric):
    def __init__(self, records, catmap):
        super().__init__()
        self._coco_ds = coco_api_from_records(records, catmap)
        self._coco_evaluator = None

    def step(self, model, xb, yb, preds):
        # TODO: Implement batch_to_cpu helper function
        self.model = model
        preds = [{k: v.to(torch.device("cpu")) for k, v in p.items()} for p in preds]
        res = {y["image_id"].item(): pred for y, pred in zip(yb, preds)}
        self.coco_evaluator.update(res)

    def end(self, model, outs):
        self.model = model
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        self._new_coco_evaluator()

    @property
    def coco_evaluator(self):
        if self._coco_evaluator is None:
            self._new_coco_evaluator()
        return self._coco_evaluator

    def _new_coco_evaluator(self):
        self._coco_evaluator = CocoEvaluator(self._coco_ds, _get_iou_types(self.model))
