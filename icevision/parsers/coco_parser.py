__all__ = ["coco", "COCOBaseParser", "COCOBBoxParser", "COCOMaskParser"]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.parsers import *


def coco(
    annotations_file: Union[str, Path],
    img_dir: Union[str, Path],
    mask: bool = True,
) -> Parser:
    parser_cls = COCOMaskParser if mask else COCOBBoxParser
    return parser_cls(annotations_file, img_dir)


class COCOBaseParser(
    Parser, FilepathMixin, SizeMixin, LabelsMixin, AreasMixin, IsCrowdsMixin
):
    def __init__(self, annotations_filepath, img_dir):
        self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = img_dir

        self._imageid2info = {o["id"]: o for o in self.annotations_dict["images"]}

    def __iter__(self):
        yield from self.annotations_dict["annotations"]

    def __len__(self):
        return len(self.annotations_dict["annotations"])

    def prepare(self, o):
        self._info = self._imageid2info[o["image_id"]]

    def imageid(self, o) -> int:
        return o["image_id"]

    def filepath(self, o) -> Union[str, Path]:
        return self.img_dir / self._info["file_name"]

    def image_width_height(self, o) -> Tuple[int, int]:
        return get_image_size(self.filepath(o))

    def labels(self, o) -> List[int]:
        return [o["category_id"]]

    def areas(self, o) -> List[float]:
        return [o["area"]]

    def iscrowds(self, o) -> List[bool]:
        return [o["iscrowd"]]


class COCOBBoxParser(COCOBaseParser, BBoxesMixin):
    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xywh(*o["bbox"])]


class COCOMaskParser(COCOBBoxParser, MasksMixin):
    def masks(self, o) -> List[MaskArray]:
        seg = o["segmentation"]
        if o["iscrowd"]:
            return [RLE.from_coco(seg["counts"])]
        else:
            return [Polygon(seg)]
