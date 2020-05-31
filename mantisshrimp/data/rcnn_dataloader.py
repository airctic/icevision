__all__ = ["item2rcnn_training_sample", "rcnn_collate", "RCNNDataLoader"]

from ..imports import *
from ..utils import *
from ..core import *


_fake_box = [0, 1, 2, 3]


def _iid2tensor(imageid: int):
    return tensor(imageid, dtype=torch.int64)


def _labels2tensor(labels: List[int]):
    return tensor(labels or [0], dtype=torch.int64)


def _iscrowds2tensor(vs: List[int]):
    return tensor(vs or [0], dtype=torch.uint8)


def _bboxes2tensor(bxs: List[BBox]):
    return tensor([o.xyxy for o in bxs] or [_fake_box], dtype=torch.float)


def _areas2tensor(bxs: List[BBox]):
    return tensor([o.area for o in bxs] or [4])


def _masks2tensor(masks: List[MaskArray]):
    return tensor(masks.data, dtype=torch.uint8)


def item2rcnn_training_sample(item: Item):
    x = im2tensor(item.img)
    y = {
        "image_id": tensor(item.imageid, dtype=torch.int64),
        "labels": _labels2tensor(item.labels),
        "iscrowd": _iscrowds2tensor(item.iscrowds),
        "boxes": ifnotnone(item.bboxes, _bboxes2tensor),
        "area": ifnotnone(item.bboxes, _areas2tensor),
        "masks": ifnotnone(item.masks, _masks2tensor),
        # TODO: Keypoints
    }
    return x, cleandict(y)


def rcnn_collate(items):
    ts = [item2rcnn_training_sample(o) for o in items]
    return list(zip(*ts))


RCNNDataLoader = partial(DataLoader, collate_fn=rcnn_collate)
