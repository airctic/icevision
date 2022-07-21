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
from icevision.utils import *
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfig


# utils
class TorchvisionMaskRCNNBackboneConfig(TorchvisionBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="mask_rcnn", **kwargs)


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


# backbones
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


resnet18_fpn = TorchvisionMaskRCNNBackboneConfig(backbone_fn=resnet18_fpn_fn)
resnet34_fpn = TorchvisionMaskRCNNBackboneConfig(backbone_fn=resnet34_fpn_fn)
resnet50_fpn = TorchvisionMaskRCNNBackboneConfig(backbone_fn=resnet50_fpn_fn)
resnet101_fpn = TorchvisionMaskRCNNBackboneConfig(backbone_fn=resnet101_fpn_fn)
resnet152_fpn = TorchvisionMaskRCNNBackboneConfig(backbone_fn=resnet152_fpn_fn)
resnext50_32x4d_fpn = TorchvisionMaskRCNNBackboneConfig(
    backbone_fn=resnext50_32x4d_fpn_fn
)
resnext101_32x8d_fpn = TorchvisionMaskRCNNBackboneConfig(
    backbone_fn=resnext101_32x8d_fpn_fn
)
wide_resnet50_2_fpn = TorchvisionMaskRCNNBackboneConfig(
    backbone_fn=wide_resnet50_2_fpn_fn
)
wide_resnet101_2_fpn = TorchvisionMaskRCNNBackboneConfig(
    backbone_fn=wide_resnet101_2_fpn_fn
)
