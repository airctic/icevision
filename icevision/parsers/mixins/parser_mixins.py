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
    def record_mixins(self) -> List[RecordMixin]:
        return []

    def parse_fields(self, o, record) -> None:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        return []


class ImageidMixin(ParserMixin):
    """Adds `imageid` method to parser"""

    def record_mixins(self):
        return [ImageidRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        record.set_imageid(self.imageid(o))
        super().parse_fields(o, record)

    @abstractmethod
    def imageid(self, o) -> Hashable:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def imageid(self, o) -> Hashable:"]


class FilepathMixin(ParserMixin):
    """Adds `filepath` method to parser"""

    def record_mixins(self):
        return [FilepathRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        record.set_filepath(Path(self.filepath(o)))
        super().parse_fields(o, record)

    @abstractmethod
    def filepath(self, o) -> Union[str, Path]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def filepath(self, o) -> Union[str, Path]:"]


class SizeMixin(ParserMixin):
    """Adds `image_height` and `image_width` method to parser"""

    def record_mixins(self):
        return [SizeRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        # TODO: rename to image_weight, image_height
        record.set_image_size(width=self.image_width(o), height=self.image_height(o))
        super().parse_fields(o, record)

    @abstractmethod
    def image_height(self, o) -> int:
        pass

    @abstractmethod
    def image_width(self, o) -> int:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + [
            "def image_height(self, o) -> int:",
            "def image_width(self, o) -> int:",
        ]


### Annotation parsers ###
class LabelsMixin(ParserMixin):
    """Adds `labels` method to parser"""

    def record_mixins(self) -> List[RecordMixin]:
        return [LabelsRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        record.add_labels(self.labels(o))
        super().parse_fields(o, record)

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

    def record_mixins(self) -> List[RecordMixin]:
        return [BBoxesRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        record.add_bboxes(self.bboxes(o))
        super().parse_fields(o, record)

    @abstractmethod
    def bboxes(self, o) -> List[BBox]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def bboxes(self, o) -> List[BBox]:"]


class MasksMixin(ParserMixin):
    """Adds `masks` method to parser"""

    def record_mixins(self) -> List[RecordMixin]:
        return [MasksRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record) -> None:
        record.add_masks(self.masks(o))
        super().parse_fields(o, record)

    @abstractmethod
    def masks(self, o) -> List[Mask]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def masks(self, o) -> List[Mask]:"]


class AreasMixin(ParserMixin):
    """Adds `areas` method to parser"""

    def record_mixins(self) -> List[RecordMixin]:
        return [AreasRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record) -> None:
        record.add_areas(self.areas(o))
        super().parse_fields(o, record)

    @abstractmethod
    def areas(self, o) -> List[float]:
        """Returns areas of the segmentation"""

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def areas(self, o) -> List[float]:"]


class IsCrowdsMixin(ParserMixin):
    """Adds `iscrowds` method to parser"""

    def record_mixins(self) -> List[RecordMixin]:
        return [IsCrowdsRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        record.add_iscrowds(self.iscrowds(o))
        super().parse_fields(o, record)

    @abstractmethod
    def iscrowds(self, o) -> List[bool]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def iscrowds(self, o) -> List[bool]:"]
