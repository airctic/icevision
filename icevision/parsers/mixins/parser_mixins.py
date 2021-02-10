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
    # TODO: Rename to components_cls
    @property
    def components(self):
        return [o for o in type(self).__mro__ if issubclass(o, ParserMixin)]

    def parse_fields(self, o, record) -> None:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        return []


@ClassMapComponent
class ClassMapMixin(ParserMixin):
    def parse_fields(self, o, record):
        record.set_class_map(self.class_map)
        super().parse_fields(o, record)


@ImageidComponent
class ImageidMixin(ParserMixin):
    """Adds `imageid` method to parser"""

    def parse_fields(self, o, record):
        record.set_imageid(self.imageid(o))
        super().parse_fields(o, record)

    @abstractmethod
    def imageid(self, o) -> Hashable:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        return ["def imageid(self, o) -> Hashable:"] + super()._templates()


@FilepathComponent
class FilepathMixin(ParserMixin):
    """Adds `filepath` method to parser"""

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


@SizeComponent
class SizeMixin(ParserMixin):
    """Adds `image_width_height` method to parser"""

    def parse_fields(self, o, record):
        # TODO: rename to image_weight, image_height
        width, height = self.image_width_height(o)
        # TODO: deprecate width/height and use ImgSize
        record.set_img_size(ImgSize(width=width, height=height), original=True)
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
@LabelComponent
class LabelsMixin(ParserMixin):
    """Adds `labels` method to parser"""

    def parse_fields(self, o, record):
        # super first because class_map need to be set before
        super().parse_fields(o, record)

        names = self.labels(o)
        ids = [self.class_map.get_by_name(name) for name in names]
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


@BBoxComponent
class BBoxesMixin(ParserMixin):
    """Adds `bboxes` method to parser"""

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


@MaskComponent
class MasksMixin(ParserMixin):
    """Adds `masks` method to parser"""

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


@AreaComponent
class AreasMixin(ParserMixin):
    """Adds `areas` method to parser"""

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


@IsCrowdComponent
class IsCrowdsMixin(ParserMixin):
    """Adds `iscrowds` method to parser"""

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


@KeyPointComponent
class KeyPointsMixin(ParserMixin):
    """Adds `keypoints` method to parser"""

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
