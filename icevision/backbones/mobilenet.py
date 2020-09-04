__all__ = ["mobilenet", "mobilenet_param_groups"]

from icevision.imports import *
from icevision.utils import *


def mobilenet(pretrained: bool = True):
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
