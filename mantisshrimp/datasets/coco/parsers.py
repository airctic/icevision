__all__ = ["COCOImageInfoParser", "COCOAnnotationParser"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.parsers.defaults import *
from mantisshrimp.parsers.mixins import *


class COCOImageInfoParser(DefaultImageInfoParser):
    def __init__(self, infos, img_dir):
        super().__init__()
        self.infos = infos
        self.img_dir = img_dir

    def __iter__(self):
        yield from self.infos

    def __len__(self):
        return len(self.infos)

    def imageid(self, o) -> int:
        return o["id"]

    def filepath(self, o) -> Union[str, Path]:
        return self.img_dir / o["file_name"]

    def height(self, o) -> int:
        return o["height"]

    def width(self, o) -> int:
        return o["width"]


class COCOAnnotationParser(MaskRCNNParser, AreasParserMixin, IsCrowdsParserMixin):
    def __init__(self, annotations: list):
        self.annotations = annotations

    def __iter__(self):
        yield from self.annotations

    def __len__(self):
        return len(self.annotations)

    def imageid(self, o) -> int:
        return o["image_id"]

    def labels(self, o) -> List[int]:
        return [o["category_id"]]

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xywh(*o["bbox"])]

    def areas(self, o) -> List[float]:
        return [o["area"]]

    def masks(self, o) -> List[MaskArray]:
        seg = o["segmentation"]
        if o["iscrowd"]:
            return [RLE.from_coco(seg["counts"])]
        else:
            return [Polygon(seg)]

    def iscrowds(self, o) -> List[bool]:
        return [o["iscrowd"]]
