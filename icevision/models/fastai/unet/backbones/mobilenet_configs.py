__all__ = ["mobilenet"]

from types import MethodType
from typing import List

from icevision.utils.torch_utils import check_all_model_params_in_groups2
from torch import nn
import torchvision
from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfig


# utils
class TorchvisionUNetBackboneConfig(TorchvisionBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="unet", **kwargs)


# backbones
def mobilenet_fn(pretrained: bool = True):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)

    features = model.features
    features.out_channels = 1280
    features.param_groups = MethodType(mobilenet_param_groups, features)

    return features


def mobilenet_param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    layers = []
    layers += [nn.Sequential(*model[0])]
    layers += [nn.Sequential(*model[1:3])]
    layers += [nn.Sequential(*model[3:12])]
    layers += [nn.Sequential(*model[12:])]

    param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, param_groups)

    return param_groups


mobilenet = TorchvisionUNetBackboneConfig(backbone_fn=mobilenet_fn)
