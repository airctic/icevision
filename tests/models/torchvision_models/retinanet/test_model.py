from icevision.all import *


def test_retinanet_default_param_groups():
    model = retinanet.model(num_classes=4)

    param_groups = model.param_groups()
    assert len(param_groups) == 7
