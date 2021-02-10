__all__ = [
    "RecordComponent",
    "ClassMapRecordComponent",
    "ImageidRecordComponent",
    "ImageRecordComponent",
    "FilepathRecordComponent",
    "SizeRecordComponent",
    "LabelsRecordComponent",
    "BBoxesRecordComponent",
    "MasksRecordComponent",
    "AreasRecordComponent",
    "IsCrowdsRecordComponent",
    "KeyPointsRecordComponent",
    "ScoresRecordComponent",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core.components import *
from icevision.core.bbox import *
from icevision.core.mask import *
from icevision.core.exceptions import *
from icevision.core.keypoints import *
from icevision.core.class_map import *


class RecordComponent(Component):
    # TODO: as_dict is only necessary because of backwards compatibility
    @property
    def record(self):
        return self.composite

    def as_dict(self) -> dict:
        return {}

    def _load(self) -> None:
        return

    def _unload(self) -> None:
        return

    def _num_annotations(self) -> Dict[str, int]:
        return {}

    def _autofix(self) -> Dict[str, bool]:
        return {}

    def _remove_annotation(self, i) -> None:
        return

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        return {}

    def _repr(self) -> List[str]:
        return []


@ClassMapComponent
class ClassMapRecordComponent(RecordComponent):
    def set_class_map(self, class_map: ClassMap):
        self.class_map = class_map

    def as_dict(self) -> dict:
        return {"class_map": self.class_map}


@ImageidComponent
class ImageidRecordComponent(RecordComponent):
    def set_imageid(self, imageid: int):
        self.imageid = imageid

    def _repr(self) -> List[str]:
        return [f"Image ID: {self.imageid}"]

    def as_dict(self) -> dict:
        return {"imageid": self.imageid}


# TODO: we need a way to combine filepath and image mixin
# TODO: rename to ImageArrayRecordComponent
@ImageArrayComponent
class ImageRecordComponent(RecordComponent):
    def __init__(self, composite):
        super().__init__(composite=composite)
        self.img = None

    def set_img(self, img: np.ndarray):
        self.img = img
        height, width, _ = self.img.shape
        # this should set on SizeRecordComponent
        self.record.set_img_size(ImgSize(width=width, height=height), original=True)

    def _repr(self) -> List[str]:
        return [f"Image: {self.img}"]

    def as_dict(self) -> dict:
        return {"img": self.img}


@FilepathComponent
class FilepathRecordComponent(ImageRecordComponent):
    def set_filepath(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)

    def _load(self):
        img = open_img(self.filepath)
        self.set_img(img)

    def _unload(self):
        self.img = None

    def _autofix(self) -> Dict[str, bool]:
        exists = self.filepath.exists()
        if not exists:
            raise AutofixAbort(f"File '{self.filepath}' does not exist")

        return super()._autofix()

    def _repr(self) -> List[str]:
        return [f"Filepath: {self.filepath}", *super()._repr()]

    def as_dict(self) -> dict:
        return {"filepath": self.filepath, **super().as_dict()}


@SizeComponent
class SizeRecordComponent(RecordComponent):
    def set_image_size(self, width: int, height: int):
        # TODO: use ImgSize
        self.img_size = ImgSize(width=width, height=height)
        self.width, self.height = width, height

    def set_img_size(self, size: ImgSize, original: bool = False):
        self.img_size = size
        self.width, self.height = size

        if original:
            self.original_img_size = size

    def _repr(self) -> List[str]:
        return [
            f"Image size (width, height): ({self.width}, {self.height})",
        ]

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height}

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        info = [{"img_width": self.width, "img_height": self.height}]
        return {"img_size": info}


@LabelComponent
### Annotation parsers ###
class LabelsRecordComponent(RecordComponent):
    def __init__(self, composite):
        super().__init__(composite=composite)
        self.labels: List[int] = []

    def set_labels(self, labels: Sequence[int]):
        self.labels = list(labels)

    def add_labels(self, labels: Sequence[int]):
        self.labels.extend(labels)

    def is_valid(self) -> List[bool]:
        return [True for _ in self.labels]

    def _num_annotations(self) -> Dict[str, int]:
        return {"labels": len(self.labels)}

    def _autofix(self) -> Dict[str, bool]:
        return {"labels": [True] * len(self.labels)}

    def _remove_annotation(self, i):
        self.labels.pop(i)

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        return {"labels": self.labels}

    def _repr(self) -> List[str]:
        return [f"Labels: {self.labels}"]

    def as_dict(self) -> dict:
        return {"labels": self.labels}


@BBoxComponent
class BBoxesRecordComponent(RecordComponent):
    def __init__(self, composite):
        super().__init__(composite=composite)
        self.bboxes: List[BBox] = []

    def set_bboxes(self, bboxes: Sequence[BBox]):
        self.bboxes = list(bboxes)

    def add_bboxes(self, bboxes: Sequence[BBox]):
        self.bboxes.extend(bboxes)

    def _autofix(self) -> Dict[str, bool]:
        success = []
        for bbox in self.bboxes:
            try:
                autofixed = bbox.autofix(
                    img_w=self.record.width, img_h=self.record.height
                )
                success.append(True)
            except InvalidDataError as e:
                logger.log("AUTOFIX-FAIL", "{}", str(e))
                success.append(False)

        return {"bboxes": success}

    def _num_annotations(self) -> Dict[str, int]:
        return {"bboxes": len(self.bboxes)}

    def _remove_annotation(self, i):
        self.bboxes.pop(i)

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        objects = []
        for bbox in self.bboxes:
            x, y, w, h = bbox.xywh
            objects.append(
                {
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_width": w,
                    "bbox_height": h,
                    "bbox_sqrt_area": bbox.area ** 0.5,
                    "bbox_aspect_ratio": w / h,
                }
            )

        return {"bboxes": objects}

    def _repr(self) -> List[str]:
        return [f"BBoxes: {self.bboxes}"]

    def as_dict(self) -> dict:
        return {"bboxes": self.bboxes}


@MaskComponent
class MasksRecordComponent(RecordComponent):
    def __init__(self, composite):
        super().__init__(composite=composite)
        self.masks = EncodedRLEs()

    def set_masks(self, masks: Sequence[Mask]):
        self.masks = masks

    def add_masks(self, masks: Sequence[Mask]):
        self.masks.extend(self._masks_to_erle(masks))

    def _masks_to_erle(self, masks: Sequence[Mask]) -> List[Mask]:
        width, height = self.record.img_size
        return [mask.to_erles(h=height, w=width) for mask in masks]

    def _load(self):
        self._encoded_masks = self.masks
        self.masks = MaskArray.from_masks(
            self.masks, self.record.height, self.record.width
        )

    def _unload(self):
        self.masks = self._encoded_masks

    def _num_annotations(self) -> Dict[str, int]:
        return {"masks": len(self.masks)}

    def _remove_annotation(self, i):
        self.masks.pop(i)

    def _repr(self) -> List[str]:
        return [f"Masks: {self.masks}"]

    def as_dict(self) -> dict:
        return {"masks": self.masks}


@AreaComponent
class AreasRecordComponent(RecordComponent):
    def __init__(self, composite):
        super().__init__(composite=composite)
        self.areas: List[float] = []

    def add_areas(self, areas: Sequence[float]):
        self.areas = list(areas)

    def add_areas(self, areas: Sequence[float]):
        self.areas.extend(areas)

    def _num_annotations(self) -> Dict[str, int]:
        return {"areas": len(self.areas)}

    def _remove_annotation(self, i):
        self.areas.pop(i)

    def _repr(self) -> List[str]:
        return [f"Areas: {self.areas}"]

    def as_dict(self) -> dict:
        return {"areas": self.areas}


@IsCrowdComponent
class IsCrowdsRecordComponent(RecordComponent):
    def __init__(self, composite):
        super().__init__(composite=composite)
        self.iscrowds: List[bool] = []

    def set_iscrowds(self, iscrowds: Sequence[bool]):
        self.iscrowds = list(iscrowds)

    def add_iscrowds(self, iscrowds: Sequence[bool]):
        self.iscrowds.extend(iscrowds)

    def _num_annotations(self) -> Dict[str, int]:
        return {"iscrowds": len(self.iscrowds)}

    def _remove_annotation(self, i):
        self.iscrowds.pop(i)

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        return {"iscrowds": self.iscrowds}

    def _repr(self) -> List[str]:
        return [f"Is Crowds: {self.iscrowds}"]

    def as_dict(self) -> dict:
        return {"iscrowds": self.iscrowds}


@KeyPointComponent
class KeyPointsRecordComponent(RecordComponent):
    def __init__(self, composite):
        super().__init__(composite=composite)
        self.keypoints: List[KeyPoints] = []

    def set_keypoints(self, keypoints: Sequence[KeyPoints]):
        self.keypoints = list(keypoints)

    def add_keypoints(self, keypoints: Sequence[KeyPoints]):
        self.keypoints.extend(keypoints)

    def as_dict(self) -> dict:
        return {"keypoints": self.keypoints}

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        objects = [
            {"keypoint_x": kpt.x, "keypoint_y": kpt.y, "keypoint_visible": kpt.v}
            for kpt in self.keypoints
        ]
        return {"keypoints": objects}

    def _repr(self) -> List[str]:
        return {f"KeyPoints: {self.keypoints}"}


class ScoresRecordComponent(RecordComponent):
    def set_scores(self, scores: Sequence[float]):
        self.scores = scores

    def _repr(self) -> List[str]:
        return [f"Scores: {self.scores}"]

    def as_dict(self) -> dict:
        return {"scores": self.scores}
