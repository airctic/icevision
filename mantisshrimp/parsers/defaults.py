__all__ = ["DefaultImageInfoParser", "FasterRCNNParser", "MaskRCNNParser"]

from mantisshrimp.imports import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.mixins import *


class DefaultImageInfoParser(
    Parser, FilepathParserMixin, SizeParserMixin, ABC,
):
    pass


class FasterRCNNParser(Parser, LabelsParserMixin, BBoxesParserMixin, ABC):
    pass


class MaskRCNNParser(FasterRCNNParser, MasksParserMixin, IsCrowdsParserMixin, ABC):
    pass
