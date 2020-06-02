from mantisshrimp import *
from mantisshrimp.imports import Path, json
from mantisshrimp.imports import ABC


import mantisshrimp

source = Path(mantisshrimp.__file__).parent.parent / "samples"
annots_dict = json.loads((source / "annotations.json").read())

image_info_parser = COCOImageInfoParser(annots_dict["images"], source)
infos = image_info_parser.parse()
infos_ids = set(o["imageid"] for o in infos)

coco_annotation_parser = COCOAnnotationParser2(annots_dict["annotations"])
annotations = coco_annotation_parser.parse()
annotations_ids = annotations["imageid"]

# if not annotations_ids.issubeset()

annotations_ids = {1, 2, 3}

# if not annotations_ids.issubset(infos_ids):
valid_ids = infos_ids.intersection(annotations_ids)
excluded = infos_ids.union(annotations_ids) - valid_ids
print(f"Removed {excluded}")

valid_ids

diff = annotations_ids - infos_ids
infos_ids - annotations_ids

diff
# raise RuntimeError("")

annotations_ids - infos_ids

annotations_ids
infos_ids

annotations

annotations_ids
