import pytest
from icevision.all import *
from icevision.models.ross.efficientdet.backbones import *


@pytest.mark.parametrize(
    "backbone",
    [
        tf_lite0,
        d0,
        d1,
        d2,
    ],
)
def test_efficient_det_param_groups(backbone):
    model = efficientdet.model(
        backbone=backbone(pretrained=False),
        num_classes=42,
        img_size=256,
    )

    assert len(list(model.param_groups())) == 3
