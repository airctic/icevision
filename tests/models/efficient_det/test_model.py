import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name",
    [
        "efficientdet_d0",
        "efficientdet_d1",
        "tf_efficientdet_lite0",
        "tf_efficientdet_d2",
    ],
)
def test_efficient_det_param_groups(model_name):
    model = efficientdet.model(
        model_name=model_name,
        num_classes=42,
        img_size=256,
        pretrained=False,
    )

    assert len(list(model.param_groups())) == 3
