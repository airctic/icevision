from mantisshrimp import *
from mantisshrimp.imports import Path, json
from mantisshrimp.imports import ABC


import mantisshrimp

source = Path(mantisshrimp.__file__).parent.parent / "samples"
annots_dict = json.loads((source / "annotations.json").read())

image_info_parser = COCOImageInfoParser(annots_dict["images"], source)
records = image_info_parser.parse_dicted()
records


coco_annotation_parser = COCOAnnotationParser2(annots_dict["annotations"])
annotations = coco_annotation_parser.parse_dicted()
annotations
