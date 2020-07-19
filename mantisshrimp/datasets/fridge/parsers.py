__all__ = ["parser"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.parsers import *
from mantisshrimp.datasets.voc import VocXmlParser


def parser(data_dir: Path, class_map: ClassMap):
    parser = VocXmlParser(
        annotations_dir=data_dir / "odFridgeObjects/annotations",
        images_dir=data_dir / "odFridgeObjects/images",
        class_map=class_map,
    )

    return parser
