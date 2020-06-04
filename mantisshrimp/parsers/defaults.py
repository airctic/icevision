__all__ = ["ImageInfoParser", "FasterRCNNParser", "MaskRCNNParser"]

from mantisshrimp.imports import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.mixins import *


class ImageInfoParser(
    Parser, FilepathParserMixin, SizeParserMixin, ABC,
):
    pass


class FasterRCNNParser(Parser, LabelParserMixin, BBoxParserMixin, ABC):
    pass


class MaskRCNNParser(FasterRCNNParser, MaskParserMixin, ABC):
    pass
