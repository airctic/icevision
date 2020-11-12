__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

from icevision.backbones.resnet import resnext101_32x8d
from icevision.backbones.resnet_fpn import (
    resnext50_32x4d,
    wide_resnet101_2,
    wide_resnet50_2,
)
from icevision.backbones import resnet_fpn

from torchvision.ops.feature_pyramid_network import LastLevelP6P7


def resnet18(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.resnet18(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )


def resnet34(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.resnet34(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )


def resnet50(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.resnet50(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )


def resnet101(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.resnet101(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )


def resnet152(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.resnet152(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )


def resnext50_32x4d(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.resnext50_32x4d(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )


def resnext101_32x8d(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.resnext101_32x8d(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )


def wide_resnet50_2(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.wide_resnet50_2(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )


def wide_resnet101_2(
    pretrained: bool = True,
    returned_layers=(2, 3, 4),
    extra_blocks=LastLevelP6P7(256, 256),
    **kwargs,
):
    return resnet_fpn.wide_resnet101_2(
        pretrained=pretrained,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
        **kwargs,
    )
