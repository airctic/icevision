__all__ = ["COCOMetric"]

from ...imports import *
from ...utils import *
from ...models import *
from .coco_eval import CocoEvaluator
from ..metric import *
from pycocotools.coco import COCO


def records2coco(records, catmap):
    cats = [{"id": i, "name": o.name} for i, o in catmap.i2o.items()]
    annots = defaultdict(list)
    infos = []
    i = 0
    for r in tqdm(records):
        infos.append(
            {
                "id": r.info.imageid,
                "file_name": r.info.filepath.name,
                "width": r.info.w,
                "height": r.info.h,
            }
        )
        for annot in r.annot:
            annots["id"].append(i)  # TODO: Careful with ids! when over all dataset
            annots["image_id"].append(r.info.imageid)
            annots["category_id"].append(annot.label)
            annots["bbox"].append(annot.bbox.xywh)
            annots["area"].append(annot.bbox.area)
            # TODO: for other types of masks
            if notnone(annot.mask):
                annots["segmentation"].extend(annot.mask.to_erle(r.info.h, r.info.w))
            annots["iscrowd"].append(annot.iscrowd)
            # TODO: Keypoints
            i += 1
    assert allequal(lmap(len, annots.values())), "Mismatch lenght of elements"
    annots = [{k: v[i] for k, v in annots.items()} for i in range_of(annots["id"])]
    return {"images": infos, "annotations": annots, "categories": cats}


def coco_api_from_records(records, catmap):
    coco_ds = COCO()
    coco_ds.dataset = records2coco(records, catmap)
    coco_ds.createIndex()
    return coco_ds


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
