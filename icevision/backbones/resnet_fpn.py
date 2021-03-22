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
    "param_groups",
    "patch_param_groups",
]

from icevision.imports import *
from icevision.utils import *
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def _resnet_fpn(name: str, pretrained: bool = True, **kwargs):
    model = resnet_fpn_backbone(backbone_name=name, pretrained=pretrained, **kwargs)
    patch_param_groups(model)

    return model


def resnet18_fpn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet18", pretrained=pretrained, **kwargs)


def resnet34_fpn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet34", pretrained=pretrained, **kwargs)


def resnet50_fpn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet50", pretrained=pretrained, **kwargs)


def resnet101_fpn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet101", pretrained=pretrained, **kwargs)


def resnet152_fpn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnet152", pretrained=pretrained, **kwargs)


def resnext50_32x4d_fpn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnext50_32x4d", pretrained=pretrained, **kwargs)


def resnext101_32x8d_fpn(pretrained: bool = True, **kwargs):
    return _resnet_fpn("resnext101_32x8d", pretrained=pretrained, **kwargs)


def wide_resnet50_2_fpn(pretrained: bool = False, **kwargs):
    return _resnet_fpn("wide_resnet50_2", pretrained=pretrained, **kwargs)


def wide_resnet101_2_fpn(pretrained: bool = False, **kwargs):
    return _resnet_fpn("wide_resnet101_2", pretrained=pretrained, **kwargs)


def param_groups(model: nn.Module) -> List[nn.Parameter]:
    body = model.body

    layers = []
    layers += [nn.Sequential(body.conv1, body.bn1)]
    layers += [getattr(body, l) for l in list(body) if l.startswith("layer")]
    layers += [model.fpn]

    _param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, _param_groups)

    return _param_groups


def patch_param_groups(model: nn.Module) -> None:
    model.param_groups = MethodType(param_groups, model)
