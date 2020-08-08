__all__ = ["DefaultImageInfoParser", "FasterRCNNParser", "MaskRCNNParser"]

from mantisshrimp.imports import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.mixins import *


class DefaultImageInfoParser(
    Parser, FilepathMixin, SizeMixin, ABC,
):
    """Bundles `Filepath` and `Size` mixins.
    """


class FasterRCNNParser(Parser, LabelsMixin, BBoxesMixin, ABC):
    """Parser with required mixins for Faster RCNN.
    """


class MaskRCNNParser(FasterRCNNParser, MasksMixin, IsCrowdsMixin, ABC):
    """Parser with required mixins for Mask RCNN.
    """
