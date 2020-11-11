import pytest
from icevision.all import *


@pytest.fixture
def fridge_retinanet_model() -> nn.Module:
    backbone = retinanet.backbones.resnet_fpn.resnet18(pretrained=False)
    return retinanet.model(num_classes=5, backbone=backbone)
