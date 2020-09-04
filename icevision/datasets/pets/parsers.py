__all__ = ["parser", "PetsXmlParser", "PetsMaskParser", "PetsMaskFile"]

from icevision.imports import *
from icevision.core import *
from icevision import parsers


def parser(data_dir: Path, class_map: ClassMap, mask=False):
    parser = PetsXmlParser(
        annotations_dir=data_dir / "annotations/xmls",
        images_dir=data_dir / "images",
        class_map=class_map,
    )

    if mask:
        mask_parser = PetsMaskParser(data_dir / "annotations/trimaps")
        parser = parsers.CombinedParser(parser, mask_parser)

    return parser


class PetsXmlParser(parsers.VocXmlParser):
    def labels(self, o) -> List[int]:
        name = re.findall(r"^(.*)_\d+$", o.stem)[0]
        class_id = self.class_map.get_name(name)

        # there is an image with two cats (same breed)
        num_objs = len(self._root.findall("object"))

        return [class_id] * num_objs


class PetsMaskParser(parsers.VocMaskParser):
    def masks(self, o) -> List[Mask]:
        return [PetsMaskFile(o)]


@dataclass
class PetsMaskFile(VocMaskFile):
    """Extension of `MaskFile` for Pets masks.
    Invert 0s and 1s in the mask (the background is orignally 1 in the pets masks)
    Removes the color pallete and optionally drops void pixels.

    Args:
          drop_void (bool): drops the void pixels, which should have the value 255.
    """

    def to_mask(self, h, w) -> MaskArray:
        mask = super().to_mask(h=h, w=w)
        mask.data = 1 - mask.data
        return mask
