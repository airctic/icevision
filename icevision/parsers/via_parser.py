__all__ = ["via", "VIAParseError", "VIABaseParser", "VIABBoxParser"]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.parsers import *


def via(
    annotations_file: Union[str, Path],
    img_dir: Union[str, Path],
    class_map: ClassMap,
    label_field: str = "label",
) -> Parser:
    """
    Parser for JSON annotations from the VGG Image Annotator V2.
    See (https://www.robots.ox.ac.uk/~vgg/software/via/)

    Just `polygon` and `rect` shape attribute types are supported.

    # Arguments
        annotations_file: Path to the JSON annotations file exported from VIA.
        img_dir: Path to the directory containing the referenced images.
        class_map: The ClassMap object for valid labels to retrieve from the annotations file.
        label_field: Defaults to `label`. The name of the `region_attribute` containing the label.

    # Returns
        The Parser
    """
    return VIABBoxParser(annotations_file, img_dir, class_map, label_field)


class VIAParseError(Exception):
    pass


class VIABaseParser(Parser, FilepathMixin, LabelsMixin):
    def __init__(
        self,
        annotations_filepath: Union[str, Path],
        img_dir: Union[str, Path],
        class_map: ClassMap,
        label_field: str = "label",
    ):
        self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = Path(img_dir)
        self.label_field = label_field
        super().__init__(class_map=class_map)

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

    def _get_label(self, o, region_attributes: dict) -> str:
        label = region_attributes.get(self.label_field)
        if label is None:
            raise VIAParseError(
                f"Could not find label_field [{self.label_field}] while parsing [{self.imageid(o)}]"
            )
        elif not isinstance(label, str):
            raise VIAParseError(
                f"Non-string value found in label_field [{self.label_field}] while parsing [{self.imageid(o)}]"
            )
        return label

    def labels(self, o) -> List[int]:
        labels = []
        for shape in o["regions"]:
            label = self._get_label(o, shape["region_attributes"])
            if label in self.class_map._class2id:
                labels.append(label)
        return labels


class VIABBoxParser(VIABaseParser, BBoxesMixin):
    """
    VIABBoxParser parses `polygon` and `rect` shape attribute types. Polygons
    are converted into bboxes that surround the entire shape.
    """

    def bboxes(self, o) -> List[BBox]:
        boxes = []
        for shape in o["regions"]:
            label = self._get_label(o, shape["region_attributes"])
            if label in self.class_map._class2id:
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
