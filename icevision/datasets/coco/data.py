__all__ = ["class_map"]

from icevision.imports import *
from icevision.core import *


def class_map(annotations_file, background: Optional[int] = 0) -> ClassMap:
    annotations_dict = json.loads(annotations_file.read())
    classes = [cat["name"] for cat in annotations_dict["categories"]]
    return ClassMap(classes=classes, background=background)
