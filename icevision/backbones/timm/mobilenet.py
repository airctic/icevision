__all__ = ["mobilenetv3_large_100"]

import timm
from timm.models.mobilenetv3 import *


def mobilenetv3_large_100(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv3_large_100(
        pretrained=True, features_only=True, out_indices=out_indices, **kwargs
    )
