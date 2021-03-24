import pytest
from icevision.all import *


@pytest.fixture
def fridge_retinanet_model() -> nn.Module:
    backbone = models.torchvision.retinanet.backbones.resnet18_fpn(pretrained=False)
    return retinanet.model(num_classes=5, backbone=backbone)
