import pytest
from icevision import *


@pytest.mark.parametrize("model_name", ["vgg11", "vgg16"])
def test_vgg_param_groups(model_name):
    model_fn = getattr(backbones, model_name)
    model = model_fn(pretrained=False)

    param_groups = model.param_groups()
    assert len(param_groups) == 4
    check_all_model_params_in_groups2(model, param_groups)
