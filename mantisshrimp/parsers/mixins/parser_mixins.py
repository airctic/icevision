__all__ = [
    "ImageidParserMixin",
    "FilepathParserMixin",
    "SizeParserMixin",
    "LabelParserMixin",
    "BBoxParserMixin",
    "AreaParserMixin",
    "MaskParserMixin",
    "IsCrowdParserMixin",
]

from mantisshrimp.imports import *
from mantisshrimp.core import *


class ParserMixin(ABC):
    def collect_annotation_parse_funcs(self, funcs=None):
        return funcs or {}

    def collect_info_parse_funcs(self, funcs=None):
        return funcs or {}


class ImageidParserMixin(ParserMixin):
    def collect_info_parse_funcs(self, funcs=None):
        funcs = super().collect_info_parse_funcs(funcs)
        return {"imageid": self.imageid, **funcs}

    @abstractmethod
    def imageid(self, o) -> int:
        pass


class FilepathParserMixin(ParserMixin):
    def collect_info_parse_funcs(self, funcs=None):
        funcs = super().collect_info_parse_funcs(funcs)
        return {"filepath": self.filepath, **funcs}

    @abstractmethod
    def filepath(self, o) -> Union[str, Path]:
        pass


class SizeParserMixin(ParserMixin):
    def collect_info_parse_funcs(self, funcs=None):
        funcs = super().collect_info_parse_funcs(funcs)
        return {"height": self.height, "width": self.width, **funcs}

    @abstractmethod
    def height(self, o) -> int:
        pass

    @abstractmethod
    def width(self, o) -> int:
        pass


## Annotation parsers ##
class LabelParserMixin(ParserMixin):
    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"label": self.label, **funcs}

    @abstractmethod
    def label(self, o) -> int:
        """
        Returns the label for the receive sample.

        .. important:
        If you are using a RCNN model, remember to return 0 for background
        and background only.
        """


class BBoxParserMixin(ParserMixin):
    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"bbox": self.bbox, **funcs}

    @abstractmethod
    def bbox(self, o) -> BBox:
        pass


class MaskParserMixin(ParserMixin):
    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"mask": self.mask, **funcs}

    @abstractmethod
    def mask(self, o) -> Mask:
        pass


class AreaParserMixin(ParserMixin):
    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"area": self.area, **funcs}

    @abstractmethod
    def area(self, o) -> float:
        """ Returns area of the segmentation
        """


class IsCrowdParserMixin(ParserMixin):
    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"iscrowd": self.iscrowd, **funcs}

    @abstractmethod
    def iscrowd(self, o) -> bool:
        pass
