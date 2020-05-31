__all__ = ["Item"]

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
            masks=MaskArray.from_masks(r.annot.masks, r.info.h, r.info.w)
            if r.annot.masks
            else None,
            # keypoints: TODO
        )

    def asdict(self):
        return self.__dict__

    # TODO: This creates a copy, is that necessary?
    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)
