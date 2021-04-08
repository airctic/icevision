import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name",
    ["yolov5s", "yolov5m", "yolov5l"],
)
def test_yolo_model(model_name):
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=320, model_name=model_name
    )
    weights_path = get_root_dir() / "yolo" / f"{model_name}.pt"

    assert weights_path.is_file() == True
    assert len(list(model.param_groups())) == 3
    assert model.nc == 5


@pytest.mark.parametrize(
    "model_name",
    ["yolov5s", "yolov5m", "yolov5l"],
)
def test_yolo_model_notpretrained(model_name):
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=320, model_name=model_name, pretrained=False
    )

    assert len(list(model.param_groups())) == 3
    assert model.nc == 5
