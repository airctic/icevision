__all__ = [
    "resnet18_fpn",
    "resnet34_fpn",
    # "resnet50_fpn",
    # "resnet101_fpn",
    # "resnet152_fpn",
    # "resnext50_32x4d_fpn",
    # "resnext101_32x8d_fpn",
    # "wide_resnet50_2_fpn",
    # "wide_resnet101_2_fpn",
]

from icevision.imports import *
from icevision.backbones.resnet import resnet18, resnext101_32x8d
from icevision.backbones import resnet_fpn

from torchvision.ops.feature_pyramid_network import LastLevelP6P7


class BackboneConfig(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> "BackboneConfig":
        ...


class TorchvisionBackboneConfig(BackboneConfig):
    def __init__(self, backbone_fn, **backbone_fn_kwargs):
        self.backbone: nn.Module

        self.backbone_fn = backbone_fn
        self.backbone_fn_kwargs = backbone_fn_kwargs

    def __call__(self, pretrained: bool = True, **kwargs):
        self.pretrained = pretrained

        # kwargs passed to call overwrite kwargs from init
        kwargs = {**self.backbone_fn_kwargs, **kwargs}
        self.backbone = self.backbone_fn(pretrained=pretrained, **kwargs)

        return self


class RetinanetTorchvisionBackboneConfig(TorchvisionBackboneConfig):
    def __init__(
        self,
        backbone_fn,
        returned_layers=(2, 3, 4),
        extra_blocks=LastLevelP6P7(256, 256),
        **backbone_fn_kwargs,
    ):
        super().__init__(
            backbone_fn=backbone_fn,
            returned_layers=returned_layers,
            extra_blocks=extra_blocks,
            **backbone_fn_kwargs,
        )


resnet18_fpn = RetinanetTorchvisionBackboneConfig(resnet_fpn.resnet18_fpn)
resnet34_fpn = RetinanetTorchvisionBackboneConfig(resnet_fpn.resnet34_fpn)
# and so on


# def resnet18_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.resnet18_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )


# def resnet34_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.resnet34_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )


# def resnet50_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.resnet50_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )


# def resnet101_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.resnet101_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )


# def resnet152_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.resnet152_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )


# def resnext50_32x4d_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.resnext50_32x4d_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )


# def resnext101_32x8d_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.resnext101_32x8d_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )


# def wide_resnet50_2_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.wide_resnet50_2_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )


# def wide_resnet101_2_fpn(
#     pretrained: bool = True,
#     returned_layers=(2, 3, 4),
#     extra_blocks=LastLevelP6P7(256, 256),
#     **kwargs,
# ):
#     return resnet_fpn.wide_resnet101_2_fpn(
#         pretrained=pretrained,
#         returned_layers=returned_layers,
#         extra_blocks=extra_blocks,
#         **kwargs,
#     )
