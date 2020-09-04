from icevision.backbones import *
import pytest
import torch


@pytest.mark.skip
@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize(
    "backbone_name",
    [
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
        "resnext101_32x8d",
    ],
)
def test_torchvision_backbones(backbone_name, pretrained):
    model = create_torchvision_backbone(backbone=backbone_name, pretrained=pretrained)
    assert isinstance(model, torch.nn.modules.container.Sequential)
