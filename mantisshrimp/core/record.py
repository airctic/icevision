__all__ = ['ImageInfo', 'Instance', 'Annotation', 'Record']

from ..imports import *
from ..utils import *
from ..core import *
from dataclasses import replace, dataclass

@dataclass(frozen=True)
class ImageInfo:
    imageid: int
    filepath: Union[str, Path]
    h: int
    w: int
    split: int = 0

    def __post_init__(self): super().__setattr__('filepath', self.filepath)

@dataclass
class Instance:
    label: int
    bbox: BBox=None
    mask: Union[Polygon, RLE, MaskFile, MaskArray]=None
    kpts: List=None # TODO
    iscrowd: int=None

@dataclass
class Annotation:
    imageid: int
    labels: List[int]
    bboxes: List[BBox] = None
    masks: List[Polygon] = None
    kpts: List[int] = None  # TODO
    iscrowds: List[int] = None

    def __getitem__(self, i):
        bbox = ifnotnone(self.bboxes, itemgetter(i))
        # Single mask file can contain multiple masks
        mask = self.masks[i] if (notnone(self.masks) and len(self.masks) > i) else None
        kpts = ifnotnone(self.kpts, itemgetter(i))
        iscrowd = ifnotnone(self.iscrowds, itemgetter(i))
        return Instance(label=self.labels[i], bbox=bbox, mask=mask, kpts=None, iscrowd=iscrowd)  # kpts None

    def get_mask(self, h, w):
        return MaskArray.from_masks(self.masks, h, w) if notnone(self.masks) else None

@dataclass
class Record:
    info: ImageInfo
    annot: Annotation

    def new(self, *, info=None, annot=None):
        info = replace(self.info, **(info or {}))
        annot = replace(self.annot, **(annot or {}))
        return replace(self, info=info, annot=annot)

