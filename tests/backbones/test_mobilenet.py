from icevision import *


def test_mobilenet_param_groups():
    model = backbones.mobilenet(pretrained=False)
    param_groups = model.param_groups()

    assert len(param_groups) == 4
    check_all_model_params_in_groups2(model, param_groups)
