import pytest
from icevision.all import *
from icevision.models.torchvision import mask_rcnn


@pytest.mark.skip
@pytest.mark.parametrize(
    "model_name,param_groups_len",
    (
        ("resnet101_fpn", 8),
        ("resnet152_fpn", 8),
        ("resnext101_32x8d_fpn", 8),
        ("wide_resnet101_2_fpn", 8),
    ),
)
def test_mask_rcnn_fpn_backbones_large(model_name, param_groups_len):
    backbone_fn = getattr(models.torchvision.mask_rcnn.backbones, model_name)
    backbone = backbone_fn(pretrained=False)

    model = mask_rcnn.model(num_classes=4, backbone=backbone)
    assert len(model.param_groups()) == param_groups_len


@pytest.mark.skip
@pytest.mark.parametrize(
    "model_name,param_groups_len",
    (
        ("resnet34_fpn", 8),
        ("resnet50_fpn", 8),
        ("resnext50_32x4d_fpn", 8),
        ("wide_resnet50_2_fpn", 8),
    ),
)
def test_mask_rcnn_fpn_backbones_medium(model_name, param_groups_len):
    backbone_fn = getattr(models.torchvision.mask_rcnn.backbones, model_name)
    backbone = backbone_fn(pretrained=False)

    model = mask_rcnn.model(num_classes=4, backbone=backbone)
    assert len(model.param_groups()) == param_groups_len


@pytest.mark.parametrize(
    "model_name,param_groups_len",
    (
        ("resnet18_fpn", 8),
    ),
)
def test_mask_rcnn_fpn_backbones_small(model_name, param_groups_len):
    backbone_fn = getattr(models.torchvision.mask_rcnn.backbones, model_name)
    backbone = backbone_fn(pretrained=False)

    model = mask_rcnn.model(num_classes=4, backbone=backbone)
    assert len(model.param_groups()) == param_groups_len
