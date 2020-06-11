from mantisshrimp.backbones import *
import pytest
import torch


def test_torchvision_backbones():
    supported_backbones = [
        "mobilenet",
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext10132x8d",
    ]
    pretrained_status = [True, False]

    for backbone in supported_backbones:
        for is_pretrained in pretrained_status:
            model = create_torchvision_backbone(
                backbone=backbone, pretrained=is_pretrained
            )
            assert isinstance(model, torch.nn.modules.container.Sequential)
