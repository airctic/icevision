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


from icevision.backbones import resnet_fpn
from icevision.models.torchvision.backbones.backbone_config import (
    TorchvisionBackboneConfig,
)

resnet18_fpn = TorchvisionBackboneConfig(resnet_fpn.resnet18_fpn)
resnet34_fpn = TorchvisionBackboneConfig(resnet_fpn.resnet34_fpn)
resnet50_fpn = TorchvisionBackboneConfig(resnet_fpn.resnet50_fpn)
resnet101_fpn = TorchvisionBackboneConfig(resnet_fpn.resnet101_fpn)
resnet152_fpn = TorchvisionBackboneConfig(resnet_fpn.resnet152_fpn)
resnext50_32x4d_fpn = TorchvisionBackboneConfig(resnet_fpn.resnext50_32x4d_fpn)
resnext101_32x8d_fpn = TorchvisionBackboneConfig(resnet_fpn.resnext101_32x8d_fpn)
wide_resnet50_2_fpn = TorchvisionBackboneConfig(resnet_fpn.wide_resnet50_2_fpn)
wide_resnet101_2_fpn = TorchvisionBackboneConfig(resnet_fpn.wide_resnet101_2_fpn)
