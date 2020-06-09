# This imports the torchivsion defined backbones
__all__ = ["create_torchvision_backbone"]

from ..imports import *


def create_torchvision_backbone(backbone: str, is_pretrained: bool):
    # These creates models from torchvision directly, it uses imagent pretrained_weights
    if backbone == "mobile_net":
        mobile_net = torchvision.models.mobilenet_v2(pretrained=is_pretrained)
        # print(mobile_net.features) # From that I got the output channels for mobilenet
        ft_backbone = mobile_net.features
        ft_backbone.out_channels = 1280
        return ft_backbone

    elif backbone == "vgg_11":
        vgg_net = torchvision.models.vgg11(pretrained=is_pretrained)
        ft_backbone = vgg_net.features
        ft_bakbone.out_channels = 512
        return ft_backbone

    elif backbone == "vgg_13":
        vgg_net = torchvision.models.vgg13(pretrained=is_pretrained)
        ft_backbone = vgg_net.features
        ft_bakbone.out_channels = 512
        return ft_backbone

    elif backbone == "vgg_16":
        vgg_net = torchvision.models.vgg16(pretrained=is_pretrained)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "vgg_19":
        vgg_net = torchvision.models.vgg19(pretrained=is_pretrained)
        ft_backbone = vgg_net.features
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "resnet_18":
        resnet_net = torchvision.models.resnet18(pretrained=is_pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "resnet_34":
        resnet_net = torchvision.models.resnet34(pretrained=is_pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 512
        return ft_backbone

    elif backbone == "resnet_50":
        resnet_net = torchvision.models.resnet50(pretrained=is_pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        return ft_backbone

    elif backbone == "resnet_101":
        resnet_net = torchvision.models.resnet101(pretrained=is_pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        return ft_backbone

    elif backbone == "resnet_152":
        resnet_net = torchvision.models.resnet152(pretrained=is_pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        return ft_backbone

    elif backbone == "resnext101_32x8d":
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=is_pretrained)
        modules = list(resnet_net.children())[:-1]
        ft_backbone = nn.Sequential(*modules)
        ft_backbone.out_channels = 2048
        return ft_backbone
        # print(ft_model)

    else:
        # print("Error Wrong unsupported Backbone")
        raise NotImplementedError("No such backbone implemented")
