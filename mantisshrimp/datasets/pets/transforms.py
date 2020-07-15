__all__ = [
    "train_albumentations_tfms_pets",
    "valid_albumentations_tfms_pets"   
]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.transforms.albu_transform import *
from mantisshrimp.utils.utils import *


def train_albumentations_tfms_pets():
    """ Composes a Pipeline of Albumentations Transforms for PETS dataset at the train stage

    Returns:
        AlbumentationsTransform: Pipeline of Albumentations Transforms
    
    
    Examples::
        >>> train_tfms = train_albumentations_tfms_pets(train=True)
        >>> train_ds = Dataset(train_records, train_tfms)

    [[https://albumentations.readthedocs.io/en/latest/_modules/albumentations/augmentations/transforms.html/|Albumentations Transforms ]]
    """
    import albumentations as A

    # ImageNet stats
    imagenet_mean, imagenet_std = IMAGENET_STATS
    
    return AlbuTransform(
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
 

def valid_albumentations_tfms_pets():
    """ Composes a Pipeline of Albumentations Transforms for PETS dataset at the train stage

    Returns:
        AlbumentationsTransform: Pipeline of Albumentations Transforms
    
    
    Examples::
        >>> valid_tfms = valid_albumentations_tfms_pets(train=false)
        >>> valid_ds = Dataset(valid_records, valid_tfms)

    [[https://albumentations.readthedocs.io/en/latest/_modules/albumentations/augmentations/transforms.html/|Albumentations Transforms ]]
    """
    import albumentations as A

    # ImageNet stats
    imagenet_mean, imagenet_std = IMAGENET_STATS
    
    return AlbuTransform(
        [
            A.LongestMaxSize(384),
            A.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]
    )
