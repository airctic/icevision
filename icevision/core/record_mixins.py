__all__ = [
    "RecordMixin",
    "ImageidRecordMixin",
    "ImageRecordMixin",
    "FilepathRecordMixin",
    "SizeRecordMixin",
    "LabelsRecordMixin",
    "BBoxesRecordMixin",
    "MasksRecordMixin",
    "AreasRecordMixin",
    "IsCrowdsRecordMixin",
]

from icevision.imports import *
from icevision.utils import *
from .bbox import *
from .mask import *
from .exceptions import *


class RecordMixin:
    # TODO: as_dict is only necessary because of backwards compatibility
    def as_dict(self) -> dict:
        return {}

    def _load(self) -> None:
        return

    def _autofix(self) -> Dict[str, bool]:
        return {}

    def _remove_annotation(self, i) -> None:
        return


class ImageidRecordMixin(RecordMixin):
    def set_imageid(self, imageid: int):
        self.imageid = imageid

    def as_dict(self) -> dict:
        return {"imageid": self.imageid, **super().as_dict()}


# TODO: we need a way to combine filepath and image mixin
class ImageRecordMixin(RecordMixin):
    def set_img(self, img: np.ndarray):
        self.img = img
        self.height, self.width, _ = self.img.shape

    def as_dict(self) -> dict:
        return {
            "img": self.img,
            "height": self.height,
            "width": self.width,
            **super().as_dict(),
        }


class FilepathRecordMixin(RecordMixin):
    def set_filepath(self, filepath: Path):
        self.filepath = filepath

    def _load(self):
        self.img = open_img(self.filepath)
        # TODO, HACK: is it correct to overwrite height and width here?
        self.height, self.width, _ = self.img.shape
        super()._load()

    def as_dict(self) -> dict:
        # return {"filepath": self.filepath, **super().as_dict()}
        # HACK: img, height, width are conditonal, use __dict__ to circumvent that
        # can be resolved once `as_dict` gets removed
        return {**self.__dict__, **super().as_dict()}


class SizeRecordMixin(RecordMixin):
    def set_image_size(self, width: int, height: int):
        self.width, self.height = width, height

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, **super().as_dict()}


### Annotation parsers ###
class LabelsRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.labels: List[int] = []

    def is_valid(self) -> List[bool]:
        return [True for _ in self.labels]

    def add_labels(self, labels: List[int]):
        self.labels.extend(labels)

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.labels.pop(i)

    def as_dict(self) -> dict:
        return {"labels": self.labels, **super().as_dict()}


class BBoxesRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.bboxes: List[BBox] = []

    def _autofix(self) -> Dict[str, bool]:
        success = []
        for bbox in self.bboxes:
            try:
                autofixed = bbox.autofix(img_w=self.width, img_h=self.height)
                success.append(True)
            except InvalidDataError as e:
                logger.info("Failed to fix: {}", bbox)
                success.append(False)

        return {"bboxes": success, **super()._autofix()}

    def add_bboxes(self, bboxes):
        self.bboxes.extend(bboxes)

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.bboxes.pop(i)

    def as_dict(self) -> dict:
        return {"bboxes": self.bboxes, **super().as_dict()}


class MasksRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.masks: List[mask] = []

    def _load(self):
        super()._load()
        self.masks = MaskArray.from_masks(self.masks, self.height, self.width)

    def add_masks(self, masks: Sequence[Mask]):
        self.masks.extend(masks)

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.masks.pop(i)

    def as_dict(self) -> dict:
        return {"masks": self.masks, **super().as_dict()}


class AreasRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.areas: List[float] = []

    def add_areas(self, areas: Sequence[float]):
        self.areas.extend(areas)

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.areas.pop(i)

    def as_dict(self) -> dict:
        return {"areas": self.areas, **super().as_dict()}


class IsCrowdsRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.iscrowds: List[bool] = []

    def add_iscrowds(self, iscrowds: Sequence[bool]):
        self.iscrowds.extend(iscrowds)

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.iscrowds.pop(i)

    def as_dict(self) -> dict:
        return {"iscrowds": self.iscrowds, **super().as_dict()}
