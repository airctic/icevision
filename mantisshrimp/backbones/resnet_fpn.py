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
    "param_groups",
]

from icevision.imports import *
from icevision.utils import *
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def _resnet_fpn(name: str, pretrained: bool = True):
    model = resnet_fpn_backbone(backbone_name=name, pretrained=pretrained)
    model.param_groups = MethodType(param_groups, model)

    return model


def resnet18(pretrained: bool = True):
    return _resnet_fpn("resnet18", pretrained=pretrained)


def resnet34(pretrained: bool = True):
    return _resnet_fpn("resnet34", pretrained=pretrained)


def resnet50(pretrained: bool = True):
    return _resnet_fpn("resnet50", pretrained=pretrained)


def resnet101(pretrained: bool = True):
    return _resnet_fpn("resnet101", pretrained=pretrained)


def resnet152(pretrained: bool = True):
    return _resnet_fpn("resnet152", pretrained=pretrained)


def resnext50_32x4d(pretrained: bool = True):
    return _resnet_fpn("resnext50_32x4d", pretrained=pretrained)


def resnext101_32x8d(pretrained: bool = True):
    return _resnet_fpn("resnext101_32x8d", pretrained=pretrained)


def wide_resnet50_2(pretrained=False):
    return _resnet_fpn("wide_resnet50_2", pretrained=pretrained)


def wide_resnet101_2(pretrained=False):
    return _resnet_fpn("wide_resnet101_2", pretrained=pretrained)


def param_groups(model: nn.Module) -> List[nn.Parameter]:
    body = model.body

    layers = []
    layers += [nn.Sequential(body.conv1, body.bn1)]
    layers += [getattr(body, l) for l in list(body) if l.startswith("layer")]
    layers += [model.fpn]

    _param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, _param_groups)

    return _param_groups
