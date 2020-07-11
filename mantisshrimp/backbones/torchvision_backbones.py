# This imports the torchivsion defined backbones
__all__ = ["create_torchvision_backbone"]

from mantisshrimp.imports import *


def mobilenet_param_groups(model: nn.Module) -> List[nn.Parameter]:
    return [list(model.parameters())]


def mobilenet(pretrained: bool = True):
    mobile_net = torchvision.models.mobilenet_v2(pretrained=pretrained)

    ft_backbone = mobile_net.features
    ft_backbone.out_channels = 1280
    ft_backbone.param_groups = mobilenet_param_groups

    return ft_backbone


def vgg_param_groups(model: nn.Module) -> List[nn.Parameter]:
    return [list(model.parameters())]


def _vgg_features(model: nn.Module):
    feats = model.features
    feats.out_channels = 512
    feats.param_groups = vgg_param_groups

    return feats


def vgg11(pretrained: bool = True):
    return _vgg_features(model=torchvision.models.vgg11(pretrained=pretrained))


def vgg13(pretrained: bool = True):
    return _vgg_features(model=torchvision.models.vgg13(pretrained=pretrained))


def vgg16(pretrained: bool = True):
    return _vgg_features(model=torchvision.models.vgg16(pretrained=pretrained))


def vgg19(pretrained: bool = True):
    return _vgg_features(model=torchvision.models.vgg19(pretrained=pretrained))


def create_torchvision_backbone(backbone: str, pretrained: bool):
    # These creates models from torchvision directly, it uses imagent pretrained_weights
    if backbone == "mobilenet":
        mobile_net = torchvision.models.mobilenet_v2(pretrained=pretrained)
        ft_backbone = mobile_net.features
        ft_backbone.out_channels = 1280
        return ft_backbone

    elif backbone == "vgg11":
        vgg_net = torchvision.models.vgg11(pretrained=pretrained)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "vgg13":
        vgg_net = torchvision.models.vgg13(pretrained=pretrained)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "vgg16":
        vgg_net = torchvision.models.vgg16(pretrained=pretrained)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "vgg19":
        vgg_net = torchvision.models.vgg19(pretrained=pretrained)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "resnet18":
        resnet_net = torchvision.models.resnet18(pretrained=pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "resnet34":
        resnet_net = torchvision.models.resnet34(pretrained=pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "resnet50":
        resnet_net = torchvision.models.resnet50(pretrained=pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        return ft_backbone

    elif backbone == "resnet101":
        resnet_net = torchvision.models.resnet101(pretrained=pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        return ft_backbone

    elif backbone == "resnet152":
        resnet_net = torchvision.models.resnet152(pretrained=pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        return ft_backbone

    elif backbone == "resnext101_32x8d":
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        return ft_backbone

    else:
        raise ValueError("No such backbone implemented in mantisshrimp")
