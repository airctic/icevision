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

from icevision.imports import *
from icevision.core import *


class ParserMixin(ABC):
    def collect_annotation_parse_funcs(self, funcs=None):
        return funcs or {}

    def collect_info_parse_funcs(self, funcs=None):
        return funcs or {}

    @classmethod
    def generate_template(cls):
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        return []


class ImageidMixin(ParserMixin):
    """Adds `imageid` method to parser"""

    def collect_info_parse_funcs(self, funcs=None):
        funcs = super().collect_info_parse_funcs(funcs)
        return {"imageid": self.imageid, **funcs}

    @abstractmethod
    def imageid(self, o) -> Hashable:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def imageid(self, o) -> Hashable:"]


class FilepathMixin(ParserMixin):
    """Adds `filepath` method to parser"""

    def collect_info_parse_funcs(self, funcs=None):
        funcs = super().collect_info_parse_funcs(funcs)
        return {"filepath": self.filepath, **funcs}

    @abstractmethod
    def filepath(self, o) -> Union[str, Path]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def filepath(self, o) -> Union[str, Path]:"]


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

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def height(self, o) -> int:", "def width(self, o) -> int:"]


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

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def labels(self, o) -> List[int]:"]


class BBoxesMixin(ParserMixin):
    """Adds `bboxes` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"bboxes": self.bboxes, **funcs}

    @abstractmethod
    def bboxes(self, o) -> List[BBox]:
        pass

    @classmethod
    def generate_template(cls):
        print("def bboxes(self, o) -> List[BBox]:")
        super().generate_template()

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def bboxes(self, o) -> List[BBox]:"]


class MasksMixin(ParserMixin):
    """Adds `masks` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"masks": self.masks, **funcs}

    @abstractmethod
    def masks(self, o) -> List[Mask]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def masks(self, o) -> List[Mask]:"]


class AreasMixin(ParserMixin):
    """Adds `areas` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"areas": self.areas, **funcs}

    @abstractmethod
    def areas(self, o) -> List[float]:
        """Returns areas of the segmentation"""

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def areas(self, o) -> List[float]:"]


class IsCrowdsMixin(ParserMixin):
    """Adds `iscrowds` method to parser"""

    def collect_annotation_parse_funcs(self, funcs=None):
        funcs = super().collect_annotation_parse_funcs(funcs)
        return {"iscrowds": self.iscrowds, **funcs}

    @abstractmethod
    def iscrowds(self, o) -> List[bool]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def iscrowds(self, o) -> List[bool]:"]
