__all__ = [
    "Parser",
    "DetectionDefaultParser",
    "MaskDefaultParser",
    "KeyPointsDefaultParser",
]

from icevision.imports import *
from icevision.core import *
from icevision import parsers


class Parser(parsers.Parser):
    def __init__(self, class_map: Optional[ClassMap] = None):
        super().__init__(self, class_map=class_map)
        self.class_map.set_background(0)


class DetectionDefaultParser(
    Parser,
    parsers.FilepathMixin,
    parsers.LabelsMixin,
    parsers.BBoxesMixin,
):
    pass


class MaskDefaultParser(
    Parser,
    parsers.FilepathMixin,
    parsers.LabelsMixin,
    parsers.BBoxesMixin,
    parsers.MasksMixin,
    parsers.IsCrowdsMixin,  # TODO: is IsCrowds really necessary?
):
    pass


class KeyPointsDefaultParser(
    Parser,
    parsers.FilepathMixin,
    parsers.LabelsMixin,
    parsers.BBoxesMixin,
    parsers.KeyPointsMixin,
):
    pass
