__all__ = ["DefaultImageInfoParser", "FasterRCNN", "MaskRCNN"]

from mantisshrimp.imports import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.mixins import *


class DefaultImageInfoParser(
    Parser, FilepathMixin, SizeMixin, ABC,
):
    """Bundles `Filepath` and `Size` mixins.
    """


class FasterRCNN(Parser, LabelsMixin, BBoxesMixin, ABC):
    """Parser with required mixins for Faster RCNN.
    """


class MaskRCNN(FasterRCNN, MasksMixin, IsCrowdsMixin, ABC):
    """Parser with required mixins for Mask RCNN.
    """
