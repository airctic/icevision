__all__ = [
    "AlbuTransform",
    "aug_tfms_albumentations",
    "config_aug_tfms_train_pets",
    "config_aug_tfms_valid_pets"
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

config_aug_tfms_train_pets = {
    "max_size": 384,
    "bbox_safe_crop": (320, 320, 0.3),
    "flip": True,
    "rotate_limit": 20,
    "rgb_shift_always": True,
    "rb_contrast": True,
    "blur_limit": (1, 3),
    "images_stats": IMAGENET_STATS
}

config_aug_tfms_valid_pets = {
    "max_size": 384,
    "images_stats": IMAGENET_STATS
}

def aug_tfms_albumentations(config_aug_tfms=config_aug_tfms_train_pets):
    albu_array=[]

    if "max_size" in config_aug_tfms: 
      max_size = config_aug_tfms["max_size"]
      albu_array.append(A.LongestMaxSize(max_size))

    if "bbox_safe_crop" in config_aug_tfms: 
      bbox_sc_h, bbox_sc_w, bbox_sc_p = config_aug_tfms["bbox_safe_crop"]
      A.RandomSizedBBoxSafeCrop(bbox_sc_h, bbox_sc_w, bbox_sc_p)

    if "flip" in config_aug_tfms: 
      flip = config_aug_tfms["flip"]
      if flip: albu_array.append(A.HorizontalFlip())

    if "rotate_limit" in config_aug_tfms: 
      rotate_limit = config_aug_tfms["rotate_limit"]
      albu_array.append(A.ShiftScaleRotate(rotate_limit=rotate_limit))

    if "rgb_shift_always" in config_aug_tfms: 
      rgb_shift_always = config_aug_tfms["rgb_shift_always"]
      albu_array.append(A.RGBShift(always_apply=rgb_shift_always))

    if "rb_contrast" in config_aug_tfms: 
      rb_contrast = config_aug_tfms["rb_contrast"]
      if rb_contrast: albu_array.append(A.RandomBrightnessContrast())

    if "blur_limit" in config_aug_tfms: 
      blur_limit_min, blur_limit_max = config_aug_tfms["blur_limit"]
      albu_array.append(A.Blur(blur_limit=(blur_limit_min, blur_limit_max)))

    if "images_stats" in config_aug_tfms:     
      images_mean, images_std = config_aug_tfms["images_stats"]
      albu_array.append(A.Normalize(mean=images_mean, std=images_std))
    
    # return AlbuTransform  
    if not(albu_array):
      return AlbuTransform(albu_array)
    else:
        return None