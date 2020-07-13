__all__ = [
    "AlbuTransform",
    "aug_tfms_train_albu",
    "aug_tfms_valid_albu"
]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.transforms.transform import *
from mantisshrimp.utils.utils import *

import albumentations as A


class AlbuTransform(Transform):
    def __init__(self, tfms):
        self.bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"])
        super().__init__(tfms=A.Compose(tfms, bbox_params=self.bbox_params))

    def apply(
        self,
        img: np.ndarray,
        labels=None,
        bboxes: List[BBox] = None,
        masks: MaskArray = None,
        iscrowds: List[int] = None,
        **kwargs
    ):
        # Substitue labels with list of idxs, so we can also filter out iscrowds in case any bboxes is removed
        # TODO: Same should be done if a masks is completely removed from the image (if bboxes is not given)
        params = {"image": img}
        params["labels"] = list(range_of(labels)) if labels is not None else []
        params["bboxes"] = [o.xyxy for o in bboxes] if bboxes is not None else []
        if masks is not None:
            params["masks"] = masks.data

        d = self.tfms(**params)

        out = {"img": d["image"]}
        if labels is not None:
            out["labels"] = [labels[i] for i in d["labels"]]
        if bboxes is not None:
            out["bboxes"] = [BBox.from_xyxy(*points) for points in d["bboxes"]]
        if masks is not None:
            out["masks"] = MaskArray(np.stack(d["masks"]))
        if iscrowds is not None:
            out["iscrowds"] = [iscrowds[i] for i in d["labels"]]
        return out


def aug_tfms_train_albu(max_size=384, bbox_safe_crop=(320, 320, 0.3), rotate_limit=20, blur_limit=(1, 3), RGBShift_always=True, images_stats=IMAGENET_STATS):
    
    bbox_sc_h, bbox_sc_w, bbox_sc_p = bbox_safe_crop
    blur_limit_min, blur_limit_max = blur_limit
    images_mean, images_std = images_stats

    return AlbuTransform(
        [
            A.LongestMaxSize(max_size),
            A.RandomSizedBBoxSafeCrop(bbox_sc_h, bbox_sc_w, bbox_sc_p),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(rotate_limit=rotate_limit),
            A.RGBShift(always_apply=RGBShift_always),
            A.RandomBrightnessContrast(),
            A.Blur(blur_limit=(blur_limit_min, blur_limit_max)),
            A.Normalize(mean=images_mean, std=images_std),
        ]
    )


def aug_tfms_valid_albu(max_size=384, images_stats=IMAGENET_STATS):
    
    images_mean, images_std = images_stats

    return AlbuTransform(
        [
            A.LongestMaxSize(max_size),
            A.Normalize(mean=images_mean, std=images_std),
        ]
    )