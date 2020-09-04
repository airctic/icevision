__all__ = ["voc", "VocXmlParser", "VocMaskParser"]

import xml.etree.ElementTree as ET
from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.parsers.parser import *
from icevision.parsers.defaults import *
from icevision.parsers.mixins import *


def voc(
    annotations_dir: Union[str, Path],
    images_dir: Union[str, Path],
    class_map: ClassMap,
    mask: bool = False,
):
    parser = VocXmlParser(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        class_map=class_map,
    )

    if mask:
        mask_parser = VocMaskParser(data_dir / "annotations/trimaps")
        parser = CombinedParser(parser, mask_parser)

    return parser


class VocXmlParser(DefaultImageInfoParser, LabelsMixin, BBoxesMixin):
    def __init__(
        self,
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
        class_map: ClassMap,
    ):
        self.images_dir = Path(images_dir)
        self.class_map = class_map

        self.annotations_dir = Path(annotations_dir)
        self.annotation_files = get_files(self.annotations_dir, extensions=[".xml"])

    def __len__(self):
        return len(self.annotation_files)

    def __iter__(self):
        yield from self.annotation_files

    def prepare(self, o):
        tree = ET.parse(str(o))
        self._root = tree.getroot()
        self._filename = self._root.find("filename").text
        self._size = self._root.find("size")

    def imageid(self, o) -> Hashable:
        return str(Path(self._filename).stem)

    def filepath(self, o) -> Union[str, Path]:
        return self.images_dir / self._filename

    def height(self, o) -> int:
        return int(self._size.find("height").text)

    def width(self, o) -> int:
        return int(self._size.find("width").text)

    def labels(self, o) -> List[int]:
        labels = []
        for object in self._root.iter("object"):
            label = object.find("name").text
            label_id = self.class_map.get_name(label)
            labels.append(label_id)

        return labels

    def bboxes(self, o) -> List[BBox]:
        def to_int(x):
            return int(float(x))

        bboxes = []
        for object in self._root.iter("object"):
            xml_bbox = object.find("bndbox")
            xmin = to_int(xml_bbox.find("xmin").text)
            ymin = to_int(xml_bbox.find("ymin").text)
            xmax = to_int(xml_bbox.find("xmax").text)
            ymax = to_int(xml_bbox.find("ymax").text)

            bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)
            bboxes.append(bbox)

        return bboxes


class VocMaskParser(Parser, ImageidMixin, MasksMixin):
    def __init__(self, masks_dir: Union[str, Path]):
        self.mask_files = get_image_files(masks_dir)

    def __len__(self):
        return len(self.mask_files)

    def __iter__(self):
        yield from self.mask_files

    def imageid(self, o) -> Hashable:
        return str(Path(o).stem)

    def masks(self, o) -> List[Mask]:
        return [VocMaskFile(o)]
