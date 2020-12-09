__all__ = ["via", "VIABaseParser", "VIABBoxParser"]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.parsers import *


def via(
    annotations_file: Union[str, Path], img_dir: Union[str, Path], class_map: ClassMap
) -> Parser:
    return VIABBoxParser(annotations_file, img_dir, class_map)


class VIABaseParser(Parser, FilepathMixin, LabelsMixin):
    def __init__(
        self,
        annotations_filepath: Union[str, Path],
        img_dir: Union[str, Path],
        cls_map: ClassMap,
    ):
        self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = Path(img_dir)
        self.cls_map = cls_map

    def __iter__(self):
        yield from self.annotations_dict.values()

    def __len__(self):
        return len(self.annotations_dict.values())

    def imageid(self, o) -> Hashable:
        return o["filename"]

    def filepath(self, o) -> Path:
        return self.img_dir / f"{o['filename']}"

    def image_width_height(self, o) -> Tuple[int, int]:
        return get_image_size(self.filepath(o))

    def labels(self, o) -> List[int]:
        labels = []
        for shape in o["regions"]:
            label = shape["region_attributes"]["label"]
            if label in self.cls_map.class2id:
                labels.append(self.cls_map.get_name(label))
        return labels


class VIABBoxParser(VIABaseParser, BBoxesMixin):
    """
    VIABBoxParser parses JSON annotations from the VGG Image Annotator V2.
    See (https://www.robots.ox.ac.uk/~vgg/software/via/)

    Just `polygon` and `rect` shape attribute types are supported. Polygons
    are converted into bboxes that surround the entire shape.
    """

    def bboxes(self, o) -> List[BBox]:
        boxes = []
        for shape in o["regions"]:
            label = shape["region_attributes"]["label"]
            if label in self.cls_map.class2id:
                shape_attr = shape["shape_attributes"]
                if shape_attr["name"] == "polygon":
                    x, y = shape_attr["all_points_x"], shape_attr["all_points_y"]
                    boxes.append(BBox.from_xyxy(min(x), min(y), max(x), max(y)))
                elif shape_attr["name"] == "rect":
                    boxes.append(
                        BBox.from_xywh(
                            shape_attr["x"],
                            shape_attr["y"],
                            shape_attr["width"],
                            shape_attr["height"],
                        )
                    )
        return boxes
