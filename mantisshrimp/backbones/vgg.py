__all__ = ["vgg_param_groups", "vgg11", "vgg13", "vgg16", "vgg19"]

from icevision.imports import *
from icevision.utils import *


def _vgg_features(model: nn.Module):
    features = model.features
    features.out_channels = 512
    features.param_groups = MethodType(vgg_param_groups, features)

    return features


def vgg_param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    layers = []
    layers += [model[:4]]
    # splits layers into 3 equally sized chunks
    for group in np.array_split(model[4:], 3):
        layers += [nn.Sequential(*group)]

    param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, param_groups)

    return param_groups


def vgg11(pretrained: bool = True):
    return _vgg_features(model=torchvision.models.vgg11(pretrained=pretrained))


def vgg13(pretrained: bool = True):
    return _vgg_features(model=torchvision.models.vgg13(pretrained=pretrained))


def vgg16(pretrained: bool = True):
    return _vgg_features(model=torchvision.models.vgg16(pretrained=pretrained))


def vgg19(pretrained: bool = True):
    return _vgg_features(model=torchvision.models.vgg19(pretrained=pretrained))
