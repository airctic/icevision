__all__ = [
    "ParserMixin",
    "ClassMapMixin",
    "ImageidMixin",
    "FilepathMixin",
    "SizeMixin",
    "LabelsMixin",
    "BBoxesMixin",
    "AreasMixin",
    "MasksMixin",
    "IsCrowdsMixin",
    "KeyPointsMixin",
]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *


class ParserMixin(ABC):
    def record_mixins(self) -> List[RecordMixin]:
        return []

    def parse_fields(self, o, record) -> None:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        return []


class ClassMapMixin(ParserMixin):
    def record_mixins(self):
        return [ClassMapRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        record.set_class_map(self.class_map)
        super().parse_fields(o, record)


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
        return ["def imageid(self, o) -> Hashable:"] + super()._templates()


class FilepathMixin(ParserMixin):
    """Adds `filepath` method to parser"""

    def record_mixins(self):
        return [FilepathRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        filepath = Path(self.filepath(o))

        if not filepath.exists():
            raise AbortParseRecord(f"File '{filepath}' does not exist")

        record.set_filepath(filepath)
        super().parse_fields(o, record)

    @abstractmethod
    def filepath(self, o) -> Union[str, Path]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        return ["def filepath(self, o) -> Union[str, Path]:"] + super()._templates()


class SizeMixin(ParserMixin):
    """Adds `image_width_height` method to parser"""

    def record_mixins(self):
        return [SizeRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        # TODO: rename to image_weight, image_height
        width, height = self.image_width_height(o)
        record.set_image_size(width=width, height=height)
        super().parse_fields(o, record)

    @abstractmethod
    def image_width_height(self, o) -> Tuple[int, int]:
        """Returns the image size (width, height)."""

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        # Dynamic template based if FilepathMixin is present
        if issubclass(cls, FilepathMixin):
            return [
                "def image_width_height(self, o) -> Tuple[int, int]:\n"
                "    return get_image_size(self.filepath(o))",
            ] + templates
        return [
            "def image_width_height(self, o) -> Tuple[int, int]:",
        ] + templates


### Annotation parsers ###
class LabelsMixin(ParserMixin):
    """Adds `labels` method to parser"""

    def record_mixins(self) -> List[RecordMixin]:
        return [LabelsRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        # super first because class_map need to be set before
        super().parse_fields(o, record)

        names = self.labels(o)
        ids = [self.class_map.get_name(name) for name in names]
        record.add_labels(ids)

    @abstractmethod
    def labels(self, o) -> List[Hashable]:
        """Returns the labels for the receive sample.

        !!! danger "Important"
        Return `BACKGROUND` (imported from icevision) for background.
        """

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def labels(self, o) -> List[Hashable]:"]


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
        super().parse_fields(o, record)
        record.add_masks(self.masks(o))

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


class KeyPointsMixin(ParserMixin):
    """Adds `keypoints` method to parser"""

    def record_mixins(self) -> List[RecordMixin]:
        return [KeyPointsRecordMixin, *super().record_mixins()]

    def parse_fields(self, o, record):
        record.add_keypoints(self.keypoints(o))
        super().parse_fields(o, record)

    @abstractmethod
    def keypoints(self, o) -> List[KeyPoints]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return templates + ["def keypoints(self, o) -> List[KeyPoints]:"]
