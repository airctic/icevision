import pytest
from icevision import *


@pytest.mark.parametrize("model_name", ["resnet34", "resnext101_32x8d"])
def test_resnet_param_groups(model_name):
    model_fn = getattr(backbones, model_name)
    model = model_fn(pretrained=False)

    param_groups = model.param_groups()
    assert len(param_groups) == 5
