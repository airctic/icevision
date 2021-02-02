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
    def set_img(self, img: np.ndarray):
        self.img = img
        # TODO: setting size here?
        self.height, self.width, _ = self.img.shape

    def _repr(self) -> List[str]:
        return [f"Image: {self.img}"]

    def as_dict(self) -> dict:
        return {
            "img": self.img,
            # TODO: setting size here?
            "height": self.height,
            "width": self.width,
        }


@FilepathComponent
class FilepathRecordComponent(RecordComponent):
    def set_filepath(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)

    def _load(self):
        self.img = open_img(self.filepath)
        # TODO, HACK: is it correct to overwrite height and width here?
        # self.height, self.width, _ = self.img.shape

    def _autofix(self) -> Dict[str, bool]:
        exists = self.filepath.exists()
        if not exists:
            raise AutofixAbort(f"File '{self.filepath}' does not exist")

        return {}

    def _repr(self) -> List[str]:
        return [f"Filepath: {self.filepath}"]

    def as_dict(self) -> dict:
        return {"filepath": self.filepath}


@SizeComponent
class SizeRecordComponent(RecordComponent):
    def set_image_size(self, width: int, height: int):
        # TODO: use ImgSize
        self.width, self.height = width, height

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

    def is_valid(self) -> List[bool]:
        return [True for _ in self.labels]

    def add_labels(self, labels: List[int]):
        self.labels.extend(labels)

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

    def add_bboxes(self, bboxes):
        self.bboxes.extend(bboxes)

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

    def _load(self):
        self.masks = MaskArray.from_masks(
            self.masks, self.record.height, self.record.width
        )

    def add_masks(self, masks: Sequence[Mask]):
        erles = [
            mask.to_erles(h=self.record.height, w=self.record.width) for mask in masks
        ]
        self.masks.extend(erles)

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

    def add_keypoints(self, keypoints):
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
