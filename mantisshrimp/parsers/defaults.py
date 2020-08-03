__all__ = ["DefaultImageInfoParser", "FasterRCNNParser", "MaskRCNNParser"]

from mantisshrimp.imports import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.mixins import *


class DefaultImageInfoParser(
    Parser, FilepathParserMixin, SizeParserMixin, ABC,
):
    """Bundles `Filepath` and `Size` mixins.
    """


class FasterRCNNParser(Parser, LabelsParserMixin, BBoxesParserMixin, ABC):
    """Parser with required mixins for Faster RCNN.
    """


class MaskRCNNParser(FasterRCNNParser, MasksParserMixin, IsCrowdsParserMixin, ABC):
    """Parser with required mixins for Mask RCNN.
    """
