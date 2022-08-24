__all__ = [
    "RecordComponent",
    "ClassMapRecordComponent",
    "RecordIDRecordComponent",
    "ImageRecordComponent",
    "FilepathRecordComponent",
    "SizeRecordComponent",
    "BaseLabelsRecordComponent",
    "InstancesLabelsRecordComponent",
    "ClassificationLabelsRecordComponent",
    "GrayScaleRecordComponent",
    "BBoxesRecordComponent",
    "BaseMasksRecordComponent",
    "InstanceMasksRecordComponent",
    "SemanticMaskRecordComponent",
    "AreasRecordComponent",
    "IsCrowdsRecordComponent",
    "KeyPointsRecordComponent",
    "ScoresRecordComponent",
    "LossesRecordComponent",
]

from icevision.utils.imageio import open_gray_scale_image
from icevision.imports import *
from icevision.utils import *
from icevision.core.components import *
from icevision.core.bbox import *
from icevision.core.mask import *
from icevision.core.exceptions import *
from icevision.core.keypoints import *
from icevision.core.class_map import *
from icevision.core import tasks


class RecordComponent(TaskComponent):
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

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        return {}

    def _repr(self) -> List[str]:
        return []

    def builder_template(self) -> List[str]:
        return self._format_builder_template(self._builder_template())

    def _builder_template(self) -> List[str]:
        return []

    def _format_builder_template(self, lines):
        task = f".{self.task.name}." if self.task != tasks.common else "."
        return [line.format(task=task) for line in lines]

    def setup_transform(self, tfm) -> None:
        pass


class ClassMapRecordComponent(RecordComponent):
    def __init__(self, task):
        super().__init__(task=task)
        self.class_map = None

    def set_class_map(self, class_map: ClassMap):
        self.class_map = class_map

    def _repr(self) -> List[str]:
        return [f"Class Map: {self.class_map}"]

    def as_dict(self) -> dict:
        return {"class_map": self.class_map}

    def _builder_template(self) -> List[str]:
        return ["record{task}set_class_map(<ClassMap>)"]


class RecordIDRecordComponent(RecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.record_id = None

    def set_record_id(self, record_id: int):
        self.record_id = record_id

    def _repr(self) -> List[str]:
        return [f"Record ID: {self.record_id}"]

    def as_dict(self) -> dict:
        return {"record_id": self.record_id}


# TODO: we need a way to combine filepath and image mixin
class ImageRecordComponent(RecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.img = None

    def set_img(self, img: Union[PIL.Image.Image, np.ndarray]):
        self.img = image_to_numpy(img)
        self.composite.set_img_size(get_img_size_from_data(img), original=True)
        print("ImageRecordComponent::set_img")

    def _repr(self) -> List[str]:

        # TODO bugfix/1135 test
        if self.img is None:
            return [f"Img: {self.img}"]

        np_img = image_to_numpy(self.img)

        if isinstance(self.img, PIL.Image.Image):
            img_type_description = f"<PIL.Image; mode='{self.img.mode}'>"
        else:
            img_type_description = f"<np.ndarray> Image"

        ndims = len(np_img.shape)

        img_size = get_img_size_from_data(np_img)
        channels = get_number_of_channels(np_img)

        if ndims == 3:
            return [
                f"Img: {img_size.width}x{img_size.height}x{channels} {img_type_description}"
            ]
        elif ndims == 2:
            return [f"Img: {img_size.width}x{img_size.height} {img_type_description}"]
        else:
            raise ValueError(
                f"Expected image to have 2 or 3 dimensions, got {ndims} instead"
            )

    def _unload(self):
        self.img = None

    def as_dict(self) -> dict:
        return {"img": self.img}

    def setup_transform(self, tfm) -> None:
        tfm.setup_img(self)


class FilepathRecordComponent(ImageRecordComponent):
    def __init__(self, task=tasks.common, gray=False):
        super().__init__(task=task)
        self.gray = gray
        self.filepath = None

    def set_filepath(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)

    def _load(self):
        print(f"Loading '{self.filepath}'")
        if self.gray:
            img = open_gray_scale_image(self.filepath)
        else:
            img = open_img(self.filepath, gray=False)

        self.set_img(img)

    def _autofix(self) -> Dict[str, bool]:
        exists = self.filepath.exists()
        if not exists:
            raise AutofixAbort(f"File '{self.filepath}' does not exist")

        return super()._autofix()

    def _repr(self) -> List[str]:
        return [f"Filepath: {self.filepath}", *super()._repr()]

    def as_dict(self) -> dict:
        return {"filepath": self.filepath, **super().as_dict()}

    def _builder_template(self) -> List[str]:
        return ["record{task}set_filepath(<Union[str, Path]>)"]


class SizeRecordComponent(RecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.img_size = None

    def set_img_size(self, size: ImgSize, original: bool = False):
        self.img_size = size
        self.width = self.img_size.width
        self.height = self.img_size.height

        if original:
            self.original_img_size = size

    def setup_transform(self, tfm) -> None:
        tfm.setup_size(self)

    def _repr(self) -> List[str]:
        return [
            f"Image size {self.img_size}",
        ]

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height}

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        info = [{"img_width": self.width, "img_height": self.height}]
        return {"img_size": info}

    def _builder_template(self) -> List[str]:
        return ["record{task}set_img_size(<ImgSize>)"]


class GrayScaleRecordComponent(FilepathRecordComponent):
    """Overwrites the FilepathRecordComponent to load radiographic images like 16bit grayscale tiff images."""

    def _load(self):
        img = open_gray_scale_image(self.filepath)
        self.set_img(img)


### Annotation parsers ###
class BaseLabelsRecordComponent(ClassMapRecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.label_ids: List[int] = []
        self.labels: List[Hashable] = []

    # TODO: rename to labels_ids
    def set_labels_by_id(self, labels: Sequence[int]):
        self.label_ids = list(labels)
        # TODO, HACK: necessary because `Dataset.from_images` has no class_map
        if self.class_map is not None:
            self.labels = self._labels_ids_to_names(labels)

    def add_labels_by_id(self, labels: Sequence[int]):
        self.label_ids.extend(labels)
        if self.class_map is not None:
            self.labels.extend(self._labels_ids_to_names(labels))

    def set_labels(self, labels_names: Sequence[Hashable]):
        self.labels = list(labels_names)
        self.label_ids = self._labels_names_to_ids(labels_names)

    def add_labels(self, labels_names: Sequence[Hashable]):
        self.labels.extend(labels_names)
        self.label_ids.extend(self._labels_names_to_ids(labels_names))

    def is_valid(self) -> List[bool]:
        return [True for _ in self.label_ids]

    def _labels_ids_to_names(self, labels_ids):
        return [self.class_map.get_by_id(id) for id in labels_ids]

    def _labels_names_to_ids(self, labels_names):
        return [self.class_map.get_by_name(name) for name in labels_names]

    def _num_annotations(self) -> Dict[str, int]:
        return {
            "labels": len(self.label_ids),
        }

    def _autofix(self) -> Dict[str, bool]:
        return {"labels": [True] * len(self.label_ids)}

    def _remove_annotation(self, i):
        self.label_ids.pop(i)

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        return {**super()._aggregate_objects(), "labels": self.label_ids}

    def _repr(self) -> List[str]:
        return [*super()._repr(), f"Labels: {self.label_ids}"]

    def as_dict(self) -> dict:
        return {
            "labels": self.labels,
            "label_ids": self.label_ids,
        }

    def _builder_template(self) -> List[str]:
        return [
            *super()._builder_template(),
            "record{task}add_labels(<Sequence[Hashable]>)",
        ]


class InstancesLabelsRecordComponent(BaseLabelsRecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)

    def setup_transform(self, tfm) -> None:
        tfm.setup_instances_labels(self)


class ClassificationLabelsRecordComponent(BaseLabelsRecordComponent):
    def __init__(self, task=tasks.classification, is_multilabel: bool = False):
        super().__init__(task=task)
        self.is_multilabel = is_multilabel

    def _autofix(self):
        if not self.is_multilabel and len(self.labels) > 1:
            raise AutofixAbort(
                f"Expected a single label, got {len(self.labels)} instead. "
                f"If you want to do multi-label classification, initiate the record "
                f"with `is_multilabel=True`"
            )
        return super()._autofix()

    def one_hot_encoded(self) -> np.array:
        "Get labels as a one-hot encoded array"
        one_hot_labels = np.zeros(len(self.class_map))
        one_hot_labels[self.label_ids] = 1
        return one_hot_labels


class BBoxesRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        print("BBoxesRecordComponent::__init__")
        self.bboxes: List[BBox] = []

    def set_bboxes(self, bboxes: Sequence[BBox]):
        print("BBoxesRecordComponent::set_bboxes")
        self.bboxes = list(bboxes)

    def add_bboxes(self, bboxes: Sequence[BBox]):
        print("BBoxesRecordComponent::add_bboxes")
        self.bboxes.extend(bboxes)

    def _autofix(self) -> Dict[str, bool]:
        print("BBoxesRecordComponent::_autofix")

        success = []
        for bbox in self.bboxes:
            try:
                autofixed = bbox.autofix(
                    img_w=self.composite.width,
                    img_h=self.composite.height,
                    record_id=self.composite.record_id,
                )
                success.append(True)
            except InvalidDataError as e:
                autofix_log(
                    "AUTOFIX-FAIL", "{}", str(e), record_id=self.composite.record_id
                )
                success.append(False)

        return {"bboxes": success}

    def _num_annotations(self) -> Dict[str, int]:
        print("BBoxesRecordComponent::_num_annotations")

        return {"bboxes": len(self.bboxes)}

    def _remove_annotation(self, i):
        print("BBoxesRecordComponent::_remove_annotation")

        self.bboxes.pop(i)

    def _aggregate_objects(self) -> Dict[str, List[dict]]:
        print("BBoxesRecordComponent::_aggregate_objects")

        objects = []
        for bbox in self.bboxes:
            x, y, w, h = bbox.xywh
            objects.append(
                {
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_width": w,
                    "bbox_height": h,
                    "bbox_sqrt_area": bbox.area**0.5,
                    "bbox_aspect_ratio": w / h,
                }
            )

        return {"bboxes": objects}

    def _repr(self) -> List[str]:
        print("BBoxesRecordComponent::_repr")

        return [f"BBoxes: {self.bboxes}"]

    def as_dict(self) -> dict:
        print("BBoxesRecordComponent::as_dict")

        return {"bboxes": self.bboxes}

    def setup_transform(self, tfm) -> None:
        print("BBoxesRecordComponent::setup_transform")

        tfm.setup_bboxes(self)

    def _builder_template(self) -> List[str]:
        print("BBoxesRecordComponent::_builder_template")

        return ["record{task}add_bboxes(<Sequence[BBox]>)"]


class BaseMasksRecordComponent(RecordComponent):
    def __init__(self, task):
        super().__init__(task=task)
        # masks are each individual part that composes the mask
        # e.g. we can have multiple polygons, rles, etc
        self.masks = self.mask_parts = []
        self.mask_array: MaskArray = None

    def add_masks(self, masks: Sequence[Mask]):
        self.masks.extend(masks)

    def set_masks(self, masks: Sequence[Mask]):
        self.masks.clear()
        self.masks.extend(masks)

    def set_mask(self, mask: Mask):
        return self.set_masks([mask])

    def set_mask_array(self, mask_array: MaskArray):
        self.mask_array = mask_array

    def _load(self):
        print("BaseMasksRecordComponent::_load")
        print(f"self.masks len {len(self.masks)}")
        print(f"self.masks first {self.masks[0]}")
        print(f"self.composite.height {self.composite.height}")
        print(f"self.composite.width {self.composite.width}")
        mask_array = MaskArray.from_masks(
            self.masks, h=self.composite.height, w=self.composite.width
        )
        print(f"_load mask_array.shape {mask_array.shape}")
        self.set_mask_array(mask_array)

    def _unload(self):
        self.mask_array = None

    def setup_transform(self, tfm) -> None:
        tfm.setup_masks(self)

    def _repr(self) -> List[str]:
        return [f"masks: {self.masks}", f"mask_array: {self.mask_array}"]

    def as_dict(self) -> dict:
        return {"masks": self.masks, "mask_array": self.mask_array}

    def _remove_annotation(self, i):
        raise NotImplementedError(
            "_remove_annotation does not work for mask component, if you're getting "
            "this on autofix, it's probably means this record has to be fixes manually"
        )


class SemanticMaskRecordComponent(BaseMasksRecordComponent):
    def __init__(self, task=tasks.segmentation):
        super().__init__(task=task)

    def _builder_template(self) -> List[str]:
        return ["record{task}set_mask(<Mask>)"]


class InstanceMasksRecordComponent(BaseMasksRecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)

    def _builder_template(self) -> List[str]:
        return ["record{task}add_masks(<Sequence[Mask]>)"]


class AreasRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.areas: List[float] = []

    def set_areas(self, areas: Sequence[float]):
        self.areas = list(areas)

    def add_areas(self, areas: Sequence[float]):
        self.areas.extend(areas)

    def setup_transform(self, tfm) -> None:
        tfm.setup_areas(self)

    def _num_annotations(self) -> Dict[str, int]:
        return {"areas": len(self.areas)}

    def _remove_annotation(self, i):
        self.areas.pop(i)

    def _repr(self) -> List[str]:
        return [f"Areas: {self.areas}"]

    def as_dict(self) -> dict:
        return {"areas": self.areas}


class IsCrowdsRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.iscrowds: List[bool] = []

    def set_iscrowds(self, iscrowds: Sequence[bool]):
        self.iscrowds = list(iscrowds)

    def add_iscrowds(self, iscrowds: Sequence[bool]):
        self.iscrowds.extend(iscrowds)

    def setup_transform(self, tfm) -> None:
        tfm.setup_iscrowds(self)

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


class KeyPointsRecordComponent(RecordComponent):
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.keypoints: List[KeyPoints] = []

    def set_keypoints(self, keypoints: Sequence[KeyPoints]):
        self.keypoints = list(keypoints)

    def add_keypoints(self, keypoints: Sequence[KeyPoints]):
        self.keypoints.extend(keypoints)

    def setup_transform(self, tfm) -> None:
        tfm.setup_keypoints(self)

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
    def __init__(self, task=tasks.detection):
        super().__init__(task=task)
        self.scores = None

    def set_scores(self, scores: Sequence[float]):
        self.scores = scores

    def _repr(self) -> List[str]:
        return [f"Scores: {self.scores}"]

    def as_dict(self) -> dict:
        return {"scores": self.scores}


class LossesRecordComponent(RecordComponent):
    def __init__(self, task=tasks.common):
        super().__init__(task=task)
        self.losses = None

    def set_losses(self, losses: Dict):
        self.losses = losses
