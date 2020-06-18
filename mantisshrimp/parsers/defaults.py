__all__ = ["DefaultImageInfoParser", "FasterRCNNParser", "MaskRCNNParser"]

from mantisshrimp.imports import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.mixins import *


class DefaultImageInfoParser(
    Parser, FilepathParserMixin, SizeParserMixin, ABC,
):
    pass


class FasterRCNNParser(Parser, LabelParserMixin, BBoxParserMixin, ABC):
    pass


class MaskRCNNParser(FasterRCNNParser, MaskParserMixin, IsCrowdParserMixin, ABC):
    pass
