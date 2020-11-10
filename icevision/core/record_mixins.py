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
    "KeyPointsRecordMixin",
]

from icevision.imports import *
from icevision.utils import *
from .bbox import *
from .mask import *
from .exceptions import *
from .keypoints import *


class RecordMixin:
    # TODO: as_dict is only necessary because of backwards compatibility
    def as_dict(self) -> dict:
        return {}

    def _load(self) -> None:
        return

    def _num_annotations(self) -> Dict[str, int]:
        return {}

    def _autofix(self) -> Dict[str, bool]:
        return {}

    def _remove_annotation(self, i) -> None:
        return

    def _repr(self) -> List[str]:
        return []


class ImageidRecordMixin(RecordMixin):
    def set_imageid(self, imageid: int):
        self.imageid = imageid

    def _repr(self) -> List[str]:
        return [f"Image ID: {self.imageid}", *super()._repr()]

    def as_dict(self) -> dict:
        return {"imageid": self.imageid, **super().as_dict()}


# TODO: we need a way to combine filepath and image mixin
class ImageRecordMixin(RecordMixin):
    def set_img(self, img: np.ndarray):
        self.img = img
        self.height, self.width, _ = self.img.shape

    def _repr(self) -> List[str]:
        return [f"Image: {self.img}", *super()._repr()]

    def as_dict(self) -> dict:
        return {
            "img": self.img,
            "height": self.height,
            "width": self.width,
            **super().as_dict(),
        }


class FilepathRecordMixin(RecordMixin):
    def set_filepath(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)

    def _load(self):
        self.img = open_img(self.filepath)
        # TODO, HACK: is it correct to overwrite height and width here?
        self.height, self.width, _ = self.img.shape
        super()._load()

    def _autofix(self) -> Dict[str, bool]:
        exists = self.filepath.exists()
        if not exists:
            raise AutofixAbort(f"File '{self.filepath}' does not exist")

        return super()._autofix()

    def _repr(self) -> List[str]:
        return [f"Filepath: {self.filepath}", *super()._repr()]

    def as_dict(self) -> dict:
        # return {"filepath": self.filepath, **super().as_dict()}
        # HACK: img, height, width are conditonal, use __dict__ to circumvent that
        # can be resolved once `as_dict` gets removed
        return {**self.__dict__, **super().as_dict()}


class SizeRecordMixin(RecordMixin):
    def set_image_size(self, width: int, height: int):
        self.width, self.height = width, height

    def _repr(self) -> List[str]:
        return [
            f"Image size (width, height): ({self.width}, {self.height})",
            *super()._repr(),
        ]

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

    def _num_annotations(self) -> Dict[str, int]:
        return {"labels": len(self.labels), **super()._num_annotations()}

    def _autofix(self) -> Dict[str, bool]:
        return {"labels": [True] * len(self.labels), **super()._autofix()}

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.labels.pop(i)

    def _repr(self) -> List[str]:
        return [f"Labels: {self.labels}", *super()._repr()]

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
                logger.log("AUTOFIX-FAIL", "{}", str(e))
                success.append(False)

        return {"bboxes": success, **super()._autofix()}

    def add_bboxes(self, bboxes):
        self.bboxes.extend(bboxes)

    def _num_annotations(self) -> Dict[str, int]:
        return {"bboxes": len(self.bboxes), **super()._num_annotations()}

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.bboxes.pop(i)

    def _repr(self) -> List[str]:
        return [f"BBoxes: {self.bboxes}", *super()._repr()]

    def as_dict(self) -> dict:
        return {"bboxes": self.bboxes, **super().as_dict()}


class MasksRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.masks = EncodedRLEs()

    def _load(self):
        super()._load()
        self.masks = MaskArray.from_masks(self.masks, self.height, self.width)

    def add_masks(self, masks: Sequence[Mask]):
        erles = [mask.to_erles(h=self.height, w=self.width) for mask in masks]
        self.masks.extend(erles)

    def _num_annotations(self) -> Dict[str, int]:
        return {"masks": len(self.masks), **super()._num_annotations()}

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.masks.pop(i)

    def _repr(self) -> List[str]:
        return [f"Masks: {self.masks}", *super()._repr()]

    def as_dict(self) -> dict:
        return {"masks": self.masks, **super().as_dict()}


class AreasRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.areas: List[float] = []

    def add_areas(self, areas: Sequence[float]):
        self.areas.extend(areas)

    def _num_annotations(self) -> Dict[str, int]:
        return {"areas": len(self.areas), **super()._num_annotations()}

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.areas.pop(i)

    def _repr(self) -> List[str]:
        return [f"Areas: {self.areas}", *super()._repr()]

    def as_dict(self) -> dict:
        return {"areas": self.areas, **super().as_dict()}


class IsCrowdsRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.iscrowds: List[bool] = []

    def add_iscrowds(self, iscrowds: Sequence[bool]):
        self.iscrowds.extend(iscrowds)

    def _num_annotations(self) -> Dict[str, int]:
        return {"iscrowds": len(self.iscrowds), **super()._num_annotations()}

    def _remove_annotation(self, i):
        super()._remove_annotation(i)
        self.iscrowds.pop(i)

    def _repr(self) -> List[str]:
        return [f"Is Crowds: {self.iscrowds}", *super()._repr()]

    def as_dict(self) -> dict:
        return {"iscrowds": self.iscrowds, **super().as_dict()}


class KeyPointsRecordMixin(RecordMixin):
    def __init__(self):
        super().__init__()
        self.keypoints: List[KeyPoints] = []

    def add_keypoints(self, keypoints):
        self.keypoints.extend(keypoints)

    def _repr(self) -> List[str]:
        return {f"KeyPoints: {self.keypoints}", *super()._repr()}
