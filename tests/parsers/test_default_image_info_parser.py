from mantisshrimp import *
from mantisshrimp.imports import Path, json
from mantisshrimp.imports import ABC


import mantisshrimp

source = Path(mantisshrimp.__file__).parent.parent / "samples"
annots_dict = json.loads((source / "annotations.json").read())

image_info_parser = COCOImageInfoParser(annots_dict["images"], source)
records = image_info_parser.parse()
records


class COCOAnnotationParser2(
    AnnotationParser2,
    LabelParserMixin,
    BBoxParserMixin,
    MaskParserMixin,
    IsCrowdParserMixin,
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


coco_annotation_parser = COCOAnnotationParser2(annots_dict["annotations"])
annotations = coco_annotation_parser.parse()
annotations
