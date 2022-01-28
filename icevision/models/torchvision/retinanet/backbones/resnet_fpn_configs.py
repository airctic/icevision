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

from icevision.models.torchvision.faster_rcnn.backbones.resnet_fpn_utils import (
    patch_param_groups,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from icevision.models.torchvision.backbone_config import (
    TorchvisionBackboneConfig,
)


def _resnet_fpn(name: str, pretrained: bool = True, **kwargs):
    model = resnet_fpn_backbone(backbone_name=name, pretrained=pretrained, **kwargs)
    patch_param_groups(model)

    return model


def resnet18_fpn_fn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet18", pretrained=pretrained, **kwargs)


def resnet34_fpn_fn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet34", pretrained=pretrained, **kwargs)


def resnet50_fpn_fn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet50", pretrained=pretrained, **kwargs)


def resnet101_fpn_fn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet101", pretrained=pretrained, **kwargs)


def resnet152_fpn_fn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet152", pretrained=pretrained, **kwargs)


def resnext50_32x4d_fpn_fn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnext50_32x4d", pretrained=pretrained, **kwargs)


def resnext101_32x8d_fpn_fn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnext101_32x8d", pretrained=pretrained, **kwargs)


def wide_resnet50_2_fpn_fn(pretrained: bool = False, **kwargs):
    return _resnet_fpn("wide_resnet50_2", pretrained=pretrained, **kwargs)


def wide_resnet101_2_fpn_fn(pretrained: bool = False, **kwargs):
    return _resnet_fpn("wide_resnet101_2", pretrained=pretrained, **kwargs)


resnet18_fpn = TorchvisionBackboneConfig(resnet18_fpn_fn)
resnet34_fpn = TorchvisionBackboneConfig(resnet34_fpn_fn)
resnet50_fpn = TorchvisionBackboneConfig(resnet50_fpn_fn)
resnet101_fpn = TorchvisionBackboneConfig(resnet101_fpn_fn)
resnet152_fpn = TorchvisionBackboneConfig(resnet152_fpn_fn)
resnext50_32x4d_fpn = TorchvisionBackboneConfig(resnext50_32x4d_fpn_fn)
resnext101_32x8d_fpn = TorchvisionBackboneConfig(resnext101_32x8d_fpn_fn)
wide_resnet50_2_fpn = TorchvisionBackboneConfig(wide_resnet50_2_fpn_fn)
wide_resnet101_2_fpn = TorchvisionBackboneConfig(wide_resnet101_2_fpn_fn)
