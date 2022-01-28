import pytest
from icevision.all import *
from icevision.models.fastai import unet


@pytest.mark.parametrize(
    "model_name,param_groups_len",
    (
        ("mobilenet", 5),
        ("resnet18", 6),
        ("resnet34", 6),
        ("resnet50", 6),
        # ("resnet101", 6),
        # ("resnet152", 6),
        # ("resnext101_32x8d", 6),
    ),
)
def test_unet_fpn_backbones(model_name, param_groups_len):
    backbone_fn = getattr(models.fastai.unet.backbones, model_name)
    backbone = backbone_fn(pretrained=False)

    model = unet.model(img_size=64, num_classes=4, backbone=backbone)
    assert len(list(model.param_groups())) == param_groups_len
