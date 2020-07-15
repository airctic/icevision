__all__ = ["parser"]

from mantisshrimp.imports import *
from mantisshrimp.parsers import *
from mantisshrimp.datasets.fridge import CLASSES
from mantisshrimp.datasets.voc import VocXmlParser


def parser(data_dir: Path):
    parser = VocXmlParser(
        annotations_dir=data_dir / "odFridgeObjects/annotations",
        images_dir=data_dir / "odFridgeObjects/images",
        classes=CLASSES,
    )

    return parser
