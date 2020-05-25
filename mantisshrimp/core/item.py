__all__ = ['Item', 'item2tensor']

from ..imports import *
from ..utils import *
from .bbox import *
from .mask import *

@dataclass
class Item:
    img: np.ndarray
    imageid: int
    labels: List[int]
    iscrowds: List[int]
    bboxes: List[BBox] = None
    masks: MaskArray = None
    #     keypoints: #TODO

    @classmethod
    def from_record(cls, r):
        return cls(
            img=open_img(r.info.filepath),
            imageid=r.info.imageid,
            labels=r.annot.labels,
            iscrowds=r.annot.iscrowds,
            bboxes=r.annot.bboxes,
            masks=MaskArray.from_masks(r.annot.masks, r.info.h, r.info.w) if r.annot.masks else None,
            # keypoints: TODO
        )

    def asdict(self): return self.__dict__

    # TODO: This creates a copy, is that necessary?
    def replace(self, **kwargs): return dataclasses.replace(self, **kwargs)


_fake_box = [0,1,2,3]
def _iid2tensor(imageid): return tensor(imageid, dtype=torch.int64)
def _labels2tensor(labels): return tensor(labels or [0], dtype=torch.int64)
def _iscrowds2tensor(vs): return tensor(vs or [0], dtype=torch.uint8)
def _bboxes2tensor(bxs): return tensor([o.xyxy for o in bxs] or [_fake_box], dtype=torch.float)
def _areas2tensor(bxs): return tensor([o.area for o in bxs] or [4])
def _masks2tensor(masks): return tensor(masks.data, dtype=torch.uint8)

def item2tensor(item):
    x = im2tensor(item.img)
    y = {
        'image_id': tensor(item.imageid, dtype=torch.int64),
        'labels':   _labels2tensor(item.labels),
        'iscrowd':  _iscrowds2tensor(item.iscrowds),
        'boxes':    ifnotnone(item.bboxes, _bboxes2tensor),
        'area':     ifnotnone(item.bboxes, _areas2tensor),
        'masks':    ifnotnone(item.masks, _masks2tensor),
        # TODO: Keypoints
    }
    return x, cleandict(y)

