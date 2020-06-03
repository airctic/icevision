__all__ = ["COCOAnnotationParser2"]

from mantisshrimp.core import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.mixins import *


class COCOAnnotationParser2(
    Parser, LabelParserMixin, BBoxParserMixin, MaskParserMixin, IsCrowdParserMixin,
):
    def __init__(self, annotations: list):
        self.annotations = annotations

    def __iter__(self):
        yield from self.annotations

    def __len__(self):
        return len(self.annotations)

    def imageid(self, o) -> int:
        return o["image_id"]

    def label(self, o):
        return o["category_id"]

    def bbox(self, o) -> BBox:
        return BBox.from_xywh(*o["bbox"])

    def mask(self, o) -> MaskArray:
        seg = o["segmentation"]
        if o["iscrowd"]:
            return RLE.from_coco(seg["counts"])
        else:
            return Polygon(seg)

    def iscrowd(self, o) -> bool:
        return o["iscrowd"]
