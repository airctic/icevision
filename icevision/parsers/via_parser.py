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
    mask: bool = False,
    bbox_from_rect: bool = True,
    bbox_from_polygon: bool = True,
    mask_from_rect: bool = False,
    mask_from_polygon: bool = False,
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
    if mask:
        if bbox_from_rect and bbox_from_polygon:
            raise ValueError(
                f"Choose bboxes from either `rect` or `polygon` "
                "when retrieving mask annotations"
            )
        # user needs to pick at least one option for fetching masks
        # for backward compatibility, both are set to False by default
        if not mask_from_rect and not mask_from_polygon:
            raise ValueError(
                f"If fetching mask annotations, set `mask_from_rect` and/"
                "or `mask_from_polygon` to True"
            )

    parser_cls = VIAMaskParser if mask else VIABBoxParser
    return parser_cls(
        annotations_file,
        img_dir,
        class_map,
        label_field,
        bbox_from_rect,
        bbox_from_polygon,
        mask_from_rect,
        mask_from_polygon,
    )


class VIAParseError(Exception):
    pass


class VIABaseParser(Parser, FilepathMixin, LabelsMixin):
    def __init__(
        self,
        annotations_filepath: Union[str, Path],
        img_dir: Union[str, Path],
        cls_map: ClassMap,
        label_field: str = "label",
        bbox_from_rect: bool = True,
        bbox_from_polygon: bool = True,
        mask_from_rect: bool = False,
        mask_from_polygon: bool = False,
    ):
        self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = Path(img_dir)
        self.cls_map = cls_map
        self.label_field = label_field
        self.bbox_from_rect = bbox_from_rect
        self.bbox_from_polygon = bbox_from_polygon
        self.mask_from_rect = mask_from_rect
        self.mask_from_polygon = mask_from_polygon

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


def BBoxFrom(shape_attr: dict, rect: bool = False, polygon: bool = False):
    "Retrive bounding box from `rect` or `polygon` annotation"
    if rect:
        if shape_attr["name"] == "rect":
            return BBox.from_xywh(
                shape_attr["x"],
                shape_attr["y"],
                shape_attr["width"],
                shape_attr["height"],
            )
    elif polygon:
        if shape_attr["name"] == "polygon":
            x, y = shape_attr["all_points_x"], shape_attr["all_points_y"]
            return BBox.from_xyxy(min(x), min(y), max(x), max(y))
    else:
        raise VIAParseError(f"Unable to parse bbox")


def MaskFrom(
    shape_attr: dict,
    height_width: Tuple[int, int],
    rect: bool = False,
    polygon: bool = False,
):
    "Get mask annotations from `rect` or `polygon` annotations"
    if rect:
        raise NotImplementedError(f"Masks from rect are yet to be implemented")
    elif polygon:
        import PIL

        if shape_attr["name"] == "polygon":
            # create pairs of x-y points for plotting the polygon
            xy_coords = [
                (x, y)
                for x, y in zip(shape_attr["all_points_x"], shape_attr["all_points_y"])
            ]
            # create mask image from x-y coordinate pairs
            mask_img = PIL.Image.new("L", height_width, 0)
            PIL.ImageDraw.Draw(mask_img).polygon(xy_coords, outline=1, fill=1)

            # return binary mask array from created image
            mask_array = np.array(mask_img)[None, ...]
            return MaskArray(mask_array)


class VIABBoxParser(VIABaseParser, BBoxesMixin):
    """
    VIABBoxParser can parse both `polygon` and `rect` shape attribute types. Polygons
    are converted into bboxes that surround the entire shape.
    """

    def labels(self, o) -> List[int]:
        labels = []
        for shape in o["regions"]:
            shape_attr = shape["shape_attributes"]
            # only get label from `rect` shape attributes
            if self.bbox_from_rect:
                if shape_attr["name"] == "rect":
                    label = self._get_label(o, shape["region_attributes"])
                    if label in self.cls_map.class2id:
                        labels.append(self.cls_map.get_name(label))
            if self.bbox_from_polygon:
                if shape_attr["name"] == "polygon":
                    label = self._get_label(o, shape["region_attributes"])
                    if label in self.cls_map.class2id:
                        labels.append(self.cls_map.get_name(label))
        return labels

    def bboxes(self, o) -> List[BBox]:
        boxes = []
        for shape in o["regions"]:
            label = self._get_label(o, shape["region_attributes"])
            if label in self.cls_map.class2id:
                shape_attr = shape["shape_attributes"]
                if self.bbox_from_rect:
                    bbox = BBoxFrom(shape_attr, rect=True)
                    if bbox is not None:
                        boxes.append(bbox)
                if self.bbox_from_polygon:
                    bbox = BBoxFrom(shape_attr, polygon=True)
                    if bbox is not None:
                        boxes.append(bbox)
        return boxes


class VIAMaskParser(VIABBoxParser, MasksMixin):
    """
    VIAMaskParser creates a mask from the `polygon` shape attribute
    i.e. a collection of corresponding x-y coordinates
    """

    def masks(self, o) -> List[MaskArray]:
        masks = []
        for shape in o["regions"]:
            label = self._get_label(o, shape["region_attributes"])
            if label in self.cls_map.class2id:
                shape_attr = shape["shape_attributes"]
                # assume that masks exist as polygons only
                if self.mask_from_polygon:
                    mask = MaskFrom(
                        shape_attr=shape_attr,
                        height_width=get_image_size(self.filepath(o)),
                        polygon=True,
                    )
                elif self.mask_from_rect:
                    mask = MaskFrom(
                        shape_attr=shape_attr,
                        height_width=get_image_size(self.filepath(o)),
                        rect=True,
                    )
                if mask is not None:
                    masks.append(mask)
        return masks
