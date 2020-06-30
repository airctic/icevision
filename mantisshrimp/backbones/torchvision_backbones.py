# This imports the torchivsion defined backbones
__all__ = ["create_torchvision_backbone"]

from mantisshrimp.imports import *
from create_backbones import *


def create_torchvision_backbone(backbone: str, pretrained: bool):
    # These creates models from torchvision directly, it uses imagenet pretrained_weights
    if backbone == "mobilenet":
        mobile_net = torchvision.models.mobilenet_v2(pretrained=pretrained)
        # ft_backbone = mobile_net.features
        # ft_backbone.out_channels = 1280
        ft_backbone = create_backbone_features(mobile_net, 1280)
        return ft_backbone

    elif backbone == "vgg11":
        vgg_net = torchvision.models.vgg11(pretrained=pretrained)
        # ft_backbone = vgg_net.features
        # ft_backbone.out_channels = 512
        ft_backbone = create_backbone_features(vgg_net, 512)
        return ft_backbone

    elif backbone == "vgg13":
        vgg_net = torchvision.models.vgg13(pretrained=pretrained)
        # ft_backbone = vgg_net.features
        # ft_backbone.out_channels = 512
        ft_backbone = create_backbone_features(vgg_net, 512)
        return ft_backbone

    elif backbone == "vgg16":
        vgg_net = torchvision.models.vgg16(pretrained=pretrained)
        # ft_backbone = vgg_net.features
        # ft_backbone.out_channels = 512
        ft_backbone = create_backbone_features(vgg_net, 512)
        return ft_backbone

    elif backbone == "vgg19":
        vgg_net = torchvision.models.vgg19(pretrained=pretrained)
        # ft_backbone = vgg_net.features
        # ft_backbone.out_channels = 512
        ft_backbone = create_backbone_features(vgg_net, 512)
        return ft_backbone

    elif backbone == "resnet18":
        resnet_net = torchvision.models.resnet18(pretrained=pretrained)
        # modules = list(resnet_net.children())[:-1]
        # ft_backbone = nn.Sequential(*modules)
        # ft_backbone.out_channels = 512
        ft_backbone = create_backbone_adaptive(resnet_net, 512)
        return ft_backbone

    elif backbone == "resnet34":
        resnet_net = torchvision.models.resnet34(pretrained=pretrained)
        # modules = list(resnet_net.children())[:-1]
        # ft_backbone = nn.Sequential(*modules)
        # ft_backbone.out_channels = 512
        ft_backbone = create_backbone_adaptive(resnet_net, 512)
        return ft_backbone

    elif backbone == "resnet50":
        resnet_net = torchvision.models.resnet50(pretrained=pretrained)
        # modules = list(resnet_net.children())[:-1]
        # ft_backbone = nn.Sequential(*modules)
        # ft_backbone.out_channels = 2048
        ft_backbone = create_backbone_adaptive(resnet_net, 2048)
        return ft_backbone

    elif backbone == "resnet101":
        resnet_net = torchvision.models.resnet101(pretrained=pretrained)
        # modules = list(resnet_net.children())[:-1]
        # ft_backbone = nn.Sequential(*modules)
        # ft_backbone.out_channels = 2048
        ft_backbone = create_backbone_adaptive(resnet_net, 2048)
        return ft_backbone

    elif backbone == "resnet152":
        resnet_net = torchvision.models.resnet152(pretrained=pretrained)
        # modules = list(resnet_net.children())[:-1]
        # ft_backbone = nn.Sequential(*modules)
        # ft_backbone.out_channels = 2048
        ft_backbone = create_backbone_adaptive(resnet_net, 2048)
        return ft_backbone

    elif backbone == "resnext101_32x8d":
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        # modules = list(resnet_net.children())[:-1]
        # ft_backbone = nn.Sequential(*modules)
        # ft_backbone.out_channels = 2048
        ft_backbone = create_backbone_adaptive(resnet_net, 2048)
        return ft_backbone

    else:
        raise ValueError("No such backbone implemented in mantisshrimp")
