__all__ = [
    "ice_resnest14d",
    "ice_resnest26d",
    "ice_resnest50d",
    "ice_resnest50d_1s4x24d",
    "ice_resnest50d_4s2x40d",
    "ice_resnest101e",
    "ice_resnest200e",
    "ice_resnest269e",
]

from icevision.soft_dependencies import SoftDependencies

from timm.models.resnest import *


def ice_resnest14d(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return resnest14d(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_resnest26d(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return resnest26d(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_resnest50d(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return resnest50d(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_resnest50d_1s4x24d(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return resnest50d_1s4x24d(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_resnest50d_4s2x40d(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return resnest50d_4s2x40d(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_resnest101e(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return resnest101e(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_resnest200e(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return resnest200e(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )


def ice_resnest269e(pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
    return resnest269e(
        pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs
    )
