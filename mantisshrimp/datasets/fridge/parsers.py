__all__ = ["parser", "FridgeXmlParser"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.parsers import *
from mantisshrimp.datasets.fridge import CLASSES
from mantisshrimp.datasets.voc import VocXmlParser


def parser(data_dir: Path, mask=False):
    parser = FridgeXmlParser(
        annotations_dir=data_dir / "odFridgeObjects/annotations",
        images_dir=data_dir / "odFridgeObjects/images",
        classes=CLASSES,
    )

    return parser


class FridgeXmlParser(VocXmlParser):
    def labels(self, o) -> List[int]:
        name = re.findall(r"^(.*)_\d+$", o.stem)[0]
        class_id = self.category2id[name]

        return [class_id]
