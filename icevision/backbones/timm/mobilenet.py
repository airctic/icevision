__all__ = [
    "ice_mobilenetv2_100",
    "ice_mobilenetv2_110d",
    "ice_mobilenetv2_120d",
    "ice_mobilenetv2_140",
    "ice_mobilenetv3_large_075",
    "ice_mobilenetv3_large_100",
    "ice_mobilenetv3_rw",
    "ice_mobilenetv3_small_075",
    "ice_mobilenetv3_small_100",
    "ice_tf_mobilenetv3_large_075",
    "ice_tf_mobilenetv3_large_100",
    "ice_tf_mobilenetv3_large_minimal_100",
    "ice_tf_mobilenetv3_small_075",
    "ice_tf_mobilenetv3_small_100",
    "ice_tf_mobilenetv3_small_minimal_100",
]

import timm
from timm.models.mobilenetv3 import *


def ice_mobilenetv2_100(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv2_100(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_mobilenetv2_110d(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv2_110d(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_mobilenetv2_120d(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv2_120d(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_mobilenetv2_140(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv2_140(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_mobilenetv3_large_075(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv3_large_075(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_mobilenetv3_large_100(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv3_large_100(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_mobilenetv3_rw(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv3_rw(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_mobilenetv3_small_075(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv3_small_075(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_mobilenetv3_small_100(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return mobilenetv3_small_100(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_tf_mobilenetv3_large_075(
    pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs
):
    return tf_mobilenetv3_large_075(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_tf_mobilenetv3_large_100(
    pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs
):
    return tf_mobilenetv3_large_100(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_tf_mobilenetv3_large_minimal_100(
    pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs
):
    return tf_mobilenetv3_large_100(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_tf_mobilenetv3_small_075(
    pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs
):
    return tf_mobilenetv3_small_075(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_tf_mobilenetv3_small_100(
    pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs
):
    return tf_mobilenetv3_small_100(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_tf_mobilenetv3_small_minimal_100(
    pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs
):
    return tf_mobilenetv3_small_100(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )
