import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name,param_groups_len",
    (
        ("mobilenet", 5),
        ("resnet18", 6),
        ("resnet18_fpn", 7),
        ("resnext50_32x4d_fpn", 7),
        ("wide_resnet50_2_fpn", 7),
    ),
)
def test_mask_rcnn_fpn_backbones(model_name, param_groups_len):
    backbone_fn = getattr(models.torchvision.mask_rcnn.backbones, model_name)
    backbone = backbone_fn(pretrained=False)

    model = retinanet.model(num_classes=4, backbone=backbone)
    assert len(model.param_groups()) == param_groups_len


import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name,param_groups_len",
    (
        ("mobilenet", 5),
        ("resnet18", 6),
        ("resnet18_fpn", 7),
        ("resnext50_32x4d_fpn", 7),
        ("wide_resnet50_2_fpn", 7),
    ),
)
def test_mask_rcnn_fpn_backbones(model_name, param_groups_len):
    backbone_fn = getattr(models.torchvision.mask_rcnn.backbones, model_name)
    backbone = backbone_fn(pretrained=False)

    model = retinanet.model(num_classes=4, backbone=backbone)
    assert len(model.param_groups()) == param_groups_len
