__all__ = [
    "ImageidMixin",
    "FilepathMixin",
    "SizeMixin",
    "LabelsMixin",
    "BBoxesMixin",
    "AreasMixin",
    "MasksMixin",
    "IsCrowdsMixin",
]

from mantisshrimp.imports import *
from mantisshrimp.core import *


class ParserMixin(ABC):
    def collect_annotation_parse_funcs(self, funcs=None):
        return funcs or {}

    def collect_info_parse_funcs(self, funcs=None):
        return funcs or {}


class ImageidMixin(ParserMixin):
    """Adds `imageid` method to parser"""

    def collect_info_parse_funcs(self, funcs=None):
        funcs = super().collect_info_parse_funcs(funcs)
        return {"imageid": self.imageid, **funcs}

    @abstractmethod
    def imageid(self, o) -> Hashable:
        pass


class FilepathMixin(ParserMixin):
    """Adds `filepath` method to parser"""

    def collect_info_parse_funcs(self, funcs=None):
        funcs = super().collect_info_parse_funcs(funcs)
        return {"filepath": self.filepath, **funcs}

    @abstractmethod
    def filepath(self, o) -> Union[str, Path]:
        pass


class SizeMixin(ParserMixin):
    """Adds `height` and `width` method to parser"""

    def collect_info_parse_funcs(self, funcs=None):
        funcs = super().collect_info_parse_funcs(funcs)
        return {"height": self.height, "width": self.width, **funcs}

    @abstractmethod
    def height(self, o) -> int:
        pass

    @abstractmethod
    def width(self, o) -> int:
        pass


### Annotation parsers ###


class LabelsMixin(ParserMixin):
    """Adds `labels` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"labels": self.labels, **funcs}

    @abstractmethod
    def labels(self, o) -> List[int]:
        """Returns the labels for the receive sample.

        !!! danger "Important"
        If you are using a RCNN/efficientdet model,
        remember to return 0 for background.
        """


class BBoxesMixin(ParserMixin):
    """Adds `bboxes` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"bboxes": self.bboxes, **funcs}

    @abstractmethod
    def bboxes(self, o) -> List[BBox]:
        pass


class MasksMixin(ParserMixin):
    """Adds `masks` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"masks": self.masks, **funcs}

    @abstractmethod
    def masks(self, o) -> List[Mask]:
        pass


class AreasMixin(ParserMixin):
    """Adds `areas` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"areas": self.areas, **funcs}

    @abstractmethod
    def areas(self, o) -> List[float]:
        """ Returns areas of the segmentation
        """


class IsCrowdsMixin(ParserMixin):
    """Adds `iscrowds` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"iscrowds": self.iscrowds, **funcs}

    @abstractmethod
    def iscrowds(self, o) -> List[bool]:
        pass
