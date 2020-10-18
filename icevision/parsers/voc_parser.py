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
    masks_dir: Optional[Union[str, Path]] = None,
):
    if not masks_dir:
        return VocXmlParser(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            class_map=class_map,
        )
    else:
        return VocMaskParser(
            annotations_dir=annotations_dir,
            images_dir=images_dir,
            masks_dir=masks_dir,
            class_map=class_map,
        )


class VocXmlParser(Parser, FilepathMixin, SizeMixin, LabelsMixin, BBoxesMixin):
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

    def image_width_height(self, o) -> Tuple[int, int]:
        return get_image_size(self.filepath(o))

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


class VocMaskParser(VocXmlParser, MasksMixin):
    def __init__(
        self,
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        class_map: ClassMap,
    ):
        super().__init__(
            annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map
        )
        self.masks_dir = masks_dir
        self.mask_files = get_image_files(masks_dir)

        self._imageid2maskfile = {self.imageid_mask(o): o for o in self.mask_files}

        # filter annotations
        masks_ids = frozenset(self._imageid2maskfile.keys())
        self._intersection = []
        for item in super().__iter__():
            super().prepare(item)
            if super().imageid(item) in masks_ids:
                self._intersection.append(item)

    def __len__(self):
        return len(self._intersection)

    def __iter__(self):
        yield from self._intersection

    def imageid_mask(self, o) -> Hashable:
        """Should return the same as `imageid` from parent parser."""
        return str(Path(o).stem)

    def masks(self, o) -> List[Mask]:
        mask_file = self._imageid2maskfile[self.imageid(o)]
        return [VocMaskFile(mask_file)]
