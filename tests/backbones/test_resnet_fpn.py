import pytest
from icevision import *


@pytest.mark.parametrize(
    "model_name", ["resnet18_fpn", "resnext50_32x4d_fpn", "wide_resnet50_2_fpn"]
)
def test_resnet_fpn_param_groups(model_name):
    model_fn = getattr(backbones, model_name)
    model = model_fn(pretrained=False)

    param_groups = model.param_groups()
    assert len(param_groups) == 6
