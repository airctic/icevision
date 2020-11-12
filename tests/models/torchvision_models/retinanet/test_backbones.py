import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name",
    ("resnet18", "resnet34", "resnet50", "resnext50_32x4d", "wide_resnet50_2"),
)
def test_retinanet_fpn_backbones(model_name):
    backbone_fn = getattr(retinanet.backbones.resnet_fpn, model_name)
    backbone = backbone_fn(pretrained=False)

    model = retinanet.model(num_classes=4, backbone=backbone)
    assert len(model.param_groups()) == 7
