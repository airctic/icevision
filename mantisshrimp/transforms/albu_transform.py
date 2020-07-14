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


# Pets Albumentations Transforms

config_aug_tfms_train_pets = {
    """ Configuration Objects for Albumentations Transforms uring train stage

    Each dictionary key corresponds to a particular Albumentations Transform
    Each dictionary value represents some valid arguments corresponding to a specific Albumentations Transform
 
    """

    "max_size": 384,
    "bbox_safe_crop": {"height": 320, "width": 320, "p": 0.3},
    "flip": True,
    "rotate_limit": {"rotate_limit": 20},
    "rgb_shift": {"always_apply": True, "p": 0.5},
    "brightness_contrast": {"brightness_limit": 0.2, "contrast_limit": 0.2},
    "blur": {"blur_limit": (1, 3)},
    "images_stats": IMAGENET_STATS
}

config_aug_tfms_valid_pets = {
    """ Configuration Objects for Albumentations Transforms applied during validation stage

    Each dictionary key corresponds to a particular Albumentations Transform
    Each dictionary value represents some valid arguments corresponding to a specific Albumentations Transform
    """
    
    "max_size": 384,
    "images_stats": IMAGENET_STATS
}


def aug_tfms_albumentations(config_aug_tfms=config_aug_tfms_train_pets):
    """ Composes a Pipeline of Albumentations Transforms

    Args:
        config_aug_tfms: A dictionary of dictionaries. Each dictionary represents some valid arguments corresponding to a specific Albumentations Transform

    Returns:
        AlbumentationsTransform: Pipeline of Albumentations Transforms
    
    
    Examples::
        >>> train_tfms = aug_tfms_albumentations(config_aug_tfms=config_aug_tfms_train_pets)
        >>> valid_tfms = aug_tfms_albumentations(config_aug_tfms=config_aug_tfms_valid_pets)
        >>> train_ds = Dataset(train_records, train_tfms)
        >>> valid_ds = Dataset(valid_records, valid_tfms)
    """
    
    albu_array=[]

    if "max_size" in config_aug_tfms: 
      albu_array.append(A.LongestMaxSize(config_aug_tfms["max_size"]))

    if "bbox_safe_crop" in config_aug_tfms: 
      albu_array.append(A.RandomSizedBBoxSafeCrop(**config_aug_tfms["bbox_safe_crop"]))

    if "flip" in config_aug_tfms: 
      flip = config_aug_tfms["flip"]
      if flip: albu_array.append(A.HorizontalFlip())

    if "rotate_limit" in config_aug_tfms: 
      albu_array.append(A.ShiftScaleRotate(**config_aug_tfms["rotate_limit"]))

    if "rgb_shift" in config_aug_tfms:
      albu_array.append(A.RGBShift(**config_aug_tfms["rgb_shift"]))

    if "brightness_contrast" in config_aug_tfms: 
      albu_array.append(A.RandomBrightnessContrast(**config_aug_tfms["brightness_contrast"]))

    if "blur" in config_aug_tfms: 
      albu_array.append(A.Blur(**config_aug_tfms["blur"]))

    if "images_stats" in config_aug_tfms:     
      images_mean, images_std = config_aug_tfms["images_stats"]
      albu_array.append(A.Normalize(mean=images_mean, std=images_std))

    # return AlbuTransform  
    if albu_array:
      return AlbuTransform(albu_array)
    else:
        return None