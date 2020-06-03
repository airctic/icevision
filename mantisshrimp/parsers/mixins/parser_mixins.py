__all__ = [
    "ImageidParserMixin",
    "FilepathParserMixin",
    "SizeParserMixin",
    "LabelParserMixin",
    "BBoxParserMixin",
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

    # TODO: implement return type
    @abstractmethod
    def label(self, o):
        pass


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
    def mask(self, o) -> MaskArray:
        pass


class IsCrowdParserMixin(ParserMixin):
    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"iscrowd": self.iscrowd, **funcs}

    @abstractmethod
    def iscrowd(self, o) -> bool:
        pass
