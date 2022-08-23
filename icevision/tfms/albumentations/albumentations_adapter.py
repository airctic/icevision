__all__ = [
    "Adapter",
    "AlbumentationsAdapterComponent",
    "AlbumentationsImgComponent",
    "AlbumentationsSizeComponent",
    "AlbumentationsInstancesLabelsComponent",
    "AlbumentationsBBoxesComponent",
    "AlbumentationsMasksComponent",
    "AlbumentationsKeypointsComponent",
    "AlbumentationsIsCrowdsComponent",
]

import albumentations as A
from itertools import chain

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.tfms.transform import *
from icevision.tfms.albumentations.albumentations_helpers import (
    get_size_without_padding,
    get_transform,
)


@dataclass
class CollectOp:
    fn: Callable
    order: float = 0.5


class AlbumentationsAdapterComponent(Component):
    @property
    def adapter(self):
        return self.composite

    def setup(self):
        return

    def prepare(self, record):
        pass

    def collect(self, record):
        pass


class AlbumentationsImgComponent(AlbumentationsAdapterComponent):
    def setup_img(self, record):
        print("AlbumentationsImgComponent::setup_img")
        print(
            f"self.adapter._albu_in['image'] = recordd.img {image_to_numpy(record.img).shape}"
        )
        self.adapter._albu_in["image"] = image_to_numpy(record.img).transpose(
            1, 0, 2
        )  # w,h,c => h,w,c (for albumentations)

        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record):
        print("AlbumentationsImgComponent::collect")

        print(f"before record.set_img ({self.adapter._albu_out['image'].shape})")

        record.set_img(self.adapter._albu_out["image"])
        print(f"after record.set_img {record.img.shape}")


class AlbumentationsSizeComponent(AlbumentationsAdapterComponent):
    order = 0.2

    def setup_size(self, record):
        print("AlbumentationsSizeComponent::setup_size")
        self.adapter._collect_ops.append(CollectOp(self.collect, order=0.2))

    def collect(self, record) -> ImgSize:
        print("AlbumentationsSizeComponent::collect")

        img_size = self.adapter._size_no_padding
        print(f"self.adapter._size_no_padding : {img_size}")

        record.set_img_size(img_size)


class AlbumentationsInstancesLabelsComponent(AlbumentationsAdapterComponent):
    order = 0.1

    def set_labels(self, record, labels):
        print("AlbumentationsInstancesLabelsComponent::set_labels")

        # TODO HACK: Will not work for multitask, will fail silently
        print(f"Labels {labels}")
        record.detection.set_labels_by_id(labels)

    def setup_instances_labels(self, record_component):
        print("AlbumentationsInstancesLabelsComponent::setup_instances_labels")

        # TODO HACK: Will not work for multitask, will fail silently
        self._original_labels = record_component.label_ids
        # Substitue labels with list of idxs, so we can also filter out iscrowds in case any bboxes are removed
        self.adapter._albu_in["labels"] = list(range(len(self._original_labels)))

        self.adapter._collect_ops.append(CollectOp(self.collect_labels, order=0.1))

    def collect_labels(self, record):
        print("AlbumentationsInstancesLabelsComponent::collect_labels")

        self.adapter._keep_mask = np.zeros(len(self._original_labels), dtype=bool)
        self.adapter._keep_mask[self.adapter._albu_out["labels"]] = True

        labels = self.adapter._filter_attribute(self._original_labels)
        self.set_labels(record, labels)


class AlbumentationsBBoxesComponent(AlbumentationsAdapterComponent):
    def setup_bboxes(self, record_component):
        print("AlbumentationsBBoxesComponent::setup_bboxes")

        self.adapter._compose_kwargs["bbox_params"] = A.BboxParams(
            format="pascal_voc", label_fields=["labels"]
        )
        print(
            f"self.adapter._compose_kwargs['bbox_params']: {self.adapter._compose_kwargs['bbox_params']}"
        )
        # TODO: albumentations has a way of sending information that can be used for tasks

        # TODO HACK: Will not work for multitask, will fail silently
        self.adapter._albu_in["bboxes"] = [o.xyxy for o in record_component.bboxes]

        print(f"self.adapter._albu_in['bboxes'] {self.adapter._albu_in['bboxes']}")

        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record) -> List[BBox]:
        print("AlbumentationsBBoxesComponent::collect")

        print(f"self.adapter._albu_out['bboxes'] {self.adapter._albu_out['bboxes']}")

        # TODO: quickfix from 576
        # bboxes_xyxy = [_clip_bboxes(xyxy, img_h, img_w) for xyxy in d["bboxes"]]
        bboxes_xyxy = [xyxy for xyxy in self.adapter._albu_out["bboxes"]]
        bboxes = [BBox.from_xyxy(*xyxy) for xyxy in bboxes_xyxy]
        # TODO HACK: Will not work for multitask, will fail silently
        print(f"bboxes {bboxes}")
        record.detection.set_bboxes(bboxes)

    @staticmethod
    def _clip_bboxes(xyxy, h, w):
        print("AlbumentationsBBoxesComponent::_clip_bboxes")

        """Clip bboxes coordinates that are outside image dimensions."""
        x1, y1, x2, y2 = xyxy
        if w >= h:
            pad = (w - h) // 2
            h1 = pad
            h2 = w - pad
            return (x1, max(y1, h1), x2, min(y2, h2))
        else:
            pad = (h - w) // 2
            w1 = pad
            w2 = h - pad
            return (max(x1, w1), y1, min(x2, w2), y2)


class AlbumentationsMasksComponent(AlbumentationsAdapterComponent):
    def setup_masks(self, record_component):
        print("AlbumentationsMasksComponent::setup_masks")

        self._record_component = record_component
        self.adapter._albu_in["masks"] = list(record_component.mask_array.data)
        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record):
        try:
            print("AlbumentationsMasksComponent::collect")

            masks = self.adapter._filter_attribute(self.adapter._albu_out["masks"])
        except AssertionError:
            # TODO: messages should be more detailed.
            img_path = record.as_dict()["common"][
                "filepath"
            ]  # ~/.icevision/data/voc/SegmentationObject/2007_000033.png'
            data_dir = img_path.parents[1]  # ~/.icevision/data/voc'
            checklist = list(data_dir.glob(f"**/{img_path.stem}.*"))
            checklist = "".join([f"\n  -{str(path)}" for path in checklist])
            raise AttributeError(
                f"Mismatch at annotations with number of masks. Check or delete {len(checklist)} files below. {checklist}"
            )

        print(f"masks.shape {masks.shape}")
        masks = MaskArray(np.array(masks))
        self._record_component.set_mask_array(masks)
        # # set masks from the modified masks array
        # TODO: Understand whether something special needs to be done here, see comment on Github
        # Had to introduce a check for Polygon as this breaks masks behaviour
        # Do we need ot handle a case for masks?
        if all(isinstance(i, Polygon) for i in masks) or all(
            isinstance(i, Polygon) for i in self._record_component.masks
        ):
            rles = []
            for m in masks:
                if m.data.any():
                    rles.append(
                        RLE.from_coco(m.to_coco_rle(*masks.shape[1:])[0]["counts"])
                    )
            self._record_component.set_masks(rles)
        # HACK: Not sure if necessary
        self._record_component = None


class AlbumentationsKeypointsComponent(AlbumentationsAdapterComponent):
    def setup_keypoints(self, record_component):
        print("AlbumentationsKeypointsComponent::setup_keypoints")

        self.adapter._compose_kwargs["keypoint_params"] = A.KeypointParams(
            format="xy", remove_invisible=False, label_fields=["keypoints_labels"]
        )

        # not compatible with some transforms
        flat_tfms_list_ = _flatten_tfms(self.adapter.tfms_list)
        if get_transform(flat_tfms_list_, "RandomSizedBBoxSafeCrop") is not None:
            raise RuntimeError("RandomSizedBBoxSafeCrop is not supported for keypoints")

        self._kpts = record_component.keypoints
        self._kpts_xy = [xy for o in self._kpts for xy in o.xy]
        self._kpts_labels = [label for o in self._kpts for label in o.metadata.labels]
        self._kpts_visible = [visible for o in self._kpts for visible in o.visible]
        assert len(self._kpts_xy) == len(self._kpts_labels) == len(self._kpts_visible)

        self.adapter._albu_in["keypoints"] = self._kpts_xy
        self.adapter._albu_in["keypoints_labels"] = self._kpts_labels

        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record):
        print("AlbumentationsKeypointsComponent::collect")

        # remove_invisible=False, therefore all points getting in are also getting out
        assert len(self.adapter._albu_out["keypoints"]) == len(self._kpts_xy)

        tfmed_kpts = self._remove_albu_outside_keypoints(
            tfms_kpts=self.adapter._albu_out["keypoints"],
            kpts_visible=self._kpts_visible,
            size_no_padding=self.adapter._size_no_padding,
        )
        # flatten list of keypoints
        flat_kpts = list(chain.from_iterable(tfmed_kpts))
        # group keypoints from same instance
        group_kpts = [
            flat_kpts[i : i + len(flat_kpts) // len(self._kpts)]
            for i in range(
                0,
                len(flat_kpts),
                len(flat_kpts) // len(self._kpts),
            )
        ]
        assert len(group_kpts) == len(self._kpts)

        kpts = [
            KeyPoints.from_xyv(group_kpt, original_kpt.metadata)
            for group_kpt, original_kpt in zip(group_kpts, self._kpts)
        ]
        kpts = self.adapter._filter_attribute(kpts)
        record.detection.set_keypoints(kpts)

    @classmethod
    def _remove_albu_outside_keypoints(cls, tfms_kpts, kpts_visible, size_no_padding):
        """Remove keypoints that are outside image dimensions."""
        print("AlbumentationsKeypointsComponent::_remove_albu_outside_keypoints")

        v = kpts_visible
        v_n = v.copy()
        tra_n = tfms_kpts.copy()
        for i in range(len(tfms_kpts)):
            if v[i] > 0:
                v_n[i] = cls._check_kps_coords(tfms_kpts[i], size_no_padding)
                if v_n[i] == 1:
                    v_n[i] = v[i]
            if v_n[i] == 0:
                tra_n[i] = (0, 0)
            tra_n[i] = (tra_n[i][0], tra_n[i][1], v_n[i])
        return tra_n

    @staticmethod
    def _check_kps_coords(p, size_no_padding):
        print("AlbumentationsKeypointsComponent::_check_kps_coords")

        x, y = p
        w, h = size_no_padding
        if w >= h:
            pad = (w - h) // 2
            h1 = pad
            h2 = w - pad
            return int((x <= w) and (x >= 0) and (y >= h1) and (y <= h2))
        else:
            pad = (h - w) // 2
            w1 = pad
            w2 = h - pad
            return int((x <= w2) and (x >= w1) and (y >= 0) and (y <= h))


class AlbumentationsIsCrowdsComponent(AlbumentationsAdapterComponent):
    def setup_iscrowds(self, record_component):
        print("AlbumentationsIsCrowdsComponent::setup_iscrowds")

        self._iscrowds = record_component.iscrowds
        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record):
        print("AlbumentationsIsCrowdsComponent::collect")

        iscrowds = self.adapter._filter_attribute(self._iscrowds)
        record.detection.set_iscrowds(iscrowds)


class AlbumentationsAreasComponent(AlbumentationsAdapterComponent):
    def setup_areas(self, record_component):
        print("AlbumentationsAreasComponent::setup_areas")

        self._areas = record_component.areas
        self.adapter._collect_ops.append(CollectOp(self.collect))

    def collect(self, record):
        print("AlbumentationsAreasComponent::collect")

        areas = self.adapter._filter_attribute(self._areas)
        record.detection.set_areas(areas)


class Adapter(Transform, Composite):
    base_components = {
        AlbumentationsImgComponent,
        AlbumentationsSizeComponent,
        AlbumentationsInstancesLabelsComponent,
        AlbumentationsBBoxesComponent,
        AlbumentationsMasksComponent,
        AlbumentationsIsCrowdsComponent,
        AlbumentationsAreasComponent,
        AlbumentationsKeypointsComponent,
    }

    def __init__(self, tfms):
        super().__init__()
        self.tfms_list = tfms
        print(f"Adapter::tfms_list : {self.tfms_list}")

    def create_tfms(self):
        print(f"Adapter::create_tfms")

        return A.Compose(self.tfms_list, **self._compose_kwargs)

    def apply(self, record):
        print(f"Adapter::apply")

        # setup
        self._compose_kwargs = {}
        self._keep_mask = None
        self._albu_in = {}
        self._collect_ops = []
        record.setup_transform(tfm=self)

        # TODO: composing every time
        tfms = self.create_tfms()
        print("Created tfms")
        # apply transform
        print("Applying tfms...")
        self._albu_out = tfms(**self._albu_in)

        self._albu_out["image"] = self._albu_out["image"].transpose(
            1, 0, 2
        )  # h,w,c => w,h,c

        # store additional info (might be used by components on `collect`)

        img_size_no_padding = get_size_without_padding(
            tfms_list=self.tfms_list,
            before_tfm_img=record.img,
            img_size=get_img_size_from_data(self._albu_out["image"]),
        )
        print(f"Applied transform, img size without padding: {img_size_no_padding}")

        self._size_no_padding = img_size_no_padding

        # collect results
        for collect_op in sorted(self._collect_ops, key=lambda x: x.order):
            collect_op.fn(record)

        return record

    # def apply(self, record):
    #     self.prepare(record)

    #     self._albu_out = self.tfms(**self._albu_in)

    #     # store additional info (might be used by components on `collect`)
    #     self._size_no_padding = self._get_size_without_padding(record)

    #     self.reduce_on_components("collect", record=record)

    #     return record

    def _filter_attribute(self, v: list):
        print(f"Adapter::_filter_attribute")

        if self._keep_mask is None or len(self._keep_mask) == 0:
            return v
        assert len(v) == len(self._keep_mask)
        return [o for o, keep in zip(v, self._keep_mask) if keep]


def _flatten_tfms(t):
    flat = []
    for o in t:
        if _is_iter(o):
            flat += [i for i in o]
        else:
            flat.append(o)
    return flat


def _is_iter(o):
    try:
        i = iter(o)
        return True
    except:
        return False
