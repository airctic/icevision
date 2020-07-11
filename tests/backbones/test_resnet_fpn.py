import pytest
from mantisshrimp import *


@pytest.mark.parametrize(
    "model_name", ["resnet18", "resnext50_32x4d", "wide_resnet50_2"]
)
def test_resnet_fpn_param_groups(model_name):
    model_fn = getattr(backbones.resnet_fpn, model_name)
    model = model_fn(pretrained=False)

    param_groups = model.param_groups(model)
    assert len(param_groups) == 6
