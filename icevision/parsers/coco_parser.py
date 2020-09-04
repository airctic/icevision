__all__ = ["coco", "COCOImageInfoParser", "COCOBBoxParser", "COCOAnnotationParser"]

from icevision.imports import *
from icevision.core import *
from icevision.parsers import *


def coco(
    annotations_file: Union[str, Path],
    img_dir: Union[str, Path],
    mask: bool = True,
) -> ParserInterface:
    annotations_dict = json.loads(Path(annotations_file).read())

    image_info_parser = COCOImageInfoParser(
        infos=annotations_dict["images"], img_dir=img_dir
    )

    _parser_cls = COCOAnnotationParser if mask else COCOBBoxParser
    annotations_parser = _parser_cls(annotations=annotations_dict["annotations"])

    return CombinedParser(image_info_parser, annotations_parser)


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


class COCOBBoxParser(FasterRCNN, AreasMixin, IsCrowdsMixin):
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

    def iscrowds(self, o) -> List[bool]:
        return [o["iscrowd"]]


class COCOAnnotationParser(MaskRCNN, COCOBBoxParser):
    def masks(self, o) -> List[MaskArray]:
        seg = o["segmentation"]
        if o["iscrowd"]:
            return [RLE.from_coco(seg["counts"])]
        else:
            return [Polygon(seg)]
