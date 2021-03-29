__all__ = [
    "resnet18_fpn",
    "resnet34_fpn",
    "resnet50_fpn",
    "resnet101_fpn",
    "resnet152_fpn",
    "resnext50_32x4d_fpn",
    "resnext101_32x8d_fpn",
    "wide_resnet50_2_fpn",
    "wide_resnet101_2_fpn",
]

from icevision.imports import *
from icevision.backbones import resnet_fpn
from icevision.models.torchvision.backbones.backbone_config import (
    TorchvisionBackboneConfig,
)

from torchvision.ops.feature_pyramid_network import LastLevelP6P7


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
resnet50_fpn = RetinanetTorchvisionBackboneConfig(resnet_fpn.resnet50_fpn)
resnet101_fpn = RetinanetTorchvisionBackboneConfig(resnet_fpn.resnet101_fpn)
resnet152_fpn = RetinanetTorchvisionBackboneConfig(resnet_fpn.resnet152_fpn)
resnext50_32x4d_fpn = RetinanetTorchvisionBackboneConfig(resnet_fpn.resnext50_32x4d_fpn)
resnext101_32x8d_fpn = RetinanetTorchvisionBackboneConfig(
    resnet_fpn.resnext101_32x8d_fpn
)
wide_resnet50_2_fpn = RetinanetTorchvisionBackboneConfig(resnet_fpn.wide_resnet50_2_fpn)
wide_resnet101_2_fpn = RetinanetTorchvisionBackboneConfig(
    resnet_fpn.wide_resnet101_2_fpn
)
