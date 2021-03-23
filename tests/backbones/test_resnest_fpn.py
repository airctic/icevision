import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name",
    ["resnest50_fpn", "resnest101_fpn", "resnest200_fpn", "resnest269_fpn"],
)
def test_resnet_fpn_param_groups(model_name):
    model_fn = getattr(backbones, model_name)
    model = model_fn(pretrained=False)

    param_groups = model.param_groups()
    assert len(param_groups) == 6
