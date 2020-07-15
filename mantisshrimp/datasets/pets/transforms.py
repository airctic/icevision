__all__ = [
    "aug_tfms_albumentations_pets"
]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.transforms.albu_transform import *
from mantisshrimp.utils.utils import *

import albumentations as A



def aug_tfms_albumentations_pets(train=True):
    """ Composes a Pipeline of Albumentations Transforms for PETS dataset

    Args:
        train: flag to select transforms specific to either train or validation stage

    Returns:
        AlbumentationsTransform: Pipeline of Albumentations Transforms
    
    
    Examples::
        >>> train_tfms = aug_tfms_albumentations_pets(train=True)
        >>> valid_tfms = aug_tfms_albumentations_pets(train=false)
        >>> train_ds = Dataset(train_records, train_tfms)
        >>> valid_ds = Dataset(valid_records, valid_tfms)

    [[https://albumentations.readthedocs.io/en/latest/_modules/albumentations/augmentations/transforms.html/|Albumentations Transforms ]]
    """

    # ImageNet stats
    imagenet_mean, imagenet_std = IMAGENET_STATS
    
    train_tfms = AlbuTransform(
        [
            A.LongestMaxSize(384),
            A.RandomSizedBBoxSafeCrop(320, 320, p=0.3),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(rotate_limit=20),
            A.RGBShift(always_apply=True),
            A.RandomBrightnessContrast(),
            A.Blur(blur_limit=(1, 3)),
            A.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]
    )
    
    valid_tfms = AlbuTransform(
        [
            A.LongestMaxSize(384),
            A.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]
    )

    if train:
        return train_tfms
    else:
        return valid_tfms
