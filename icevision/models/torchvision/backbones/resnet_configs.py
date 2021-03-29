__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext101_32x8d",
]

from icevision.backbones import resnet
from icevision.models.torchvision.backbones.backbone_config import (
    TorchvisionBackboneConfig,
)

resnet18 = TorchvisionBackboneConfig(resnet.resnet18)
resnet34 = TorchvisionBackboneConfig(resnet.resnet34)
resnet50 = TorchvisionBackboneConfig(resnet.resnet50)
resnet101 = TorchvisionBackboneConfig(resnet.resnet101)
resnet152 = TorchvisionBackboneConfig(resnet.resnet152)
resnext101_32x8d = TorchvisionBackboneConfig(resnet.resnext101_32x8d)
