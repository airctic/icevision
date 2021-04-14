import pytest
from icevision.all import *
from icevision.models.ultralytics.yolov5.backbones import *


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large],
)
def test_yolo_model(backbone):
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=320, backbone=backbone(pretrained=True)
    )
    weights_path = get_root_dir() / "yolo" / f"{backbone.model_name}.pt"

    assert weights_path.is_file() == True
    assert len(list(model.param_groups())) == 3
    assert model.nc == 5


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large],
)
def test_yolo_model_notpretrained(backbone):
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=320, backbone=backbone(pretrained=False)
    )

    assert len(list(model.param_groups())) == 3
    assert model.nc == 5
