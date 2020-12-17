import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name", ["resnest50", "resnest101", "resnest200", "resnest269"]
)
def test_resnet_fpn_param_groups(model_name):
    model_fn = getattr(backbones.resnest_fpn, model_name)
    model = model_fn(pretrained=False)

    param_groups = model.param_groups()
    assert len(param_groups) == 6
