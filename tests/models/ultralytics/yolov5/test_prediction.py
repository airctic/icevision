import pytest
from icevision.all import *


def _test_preds(preds):
    assert len(preds) == 3
    assert isinstance(preds[0].detection.bboxes[0], BBox)
    assert len(preds[0].detection.scores) == len(preds[0].detection.labels)
    assert len(preds[2].detection.scores) == len(preds[2].detection.labels)


@pytest.mark.parametrize(
    "model_name",
    ["yolov5s", "yolov5m", "yolov5l"],
)
def test_yolo_predict(fridge_ds, model_name):
    _, valid_ds = fridge_ds
    model = models.ultralytics.yolov5.model(5, img_size=384, model_name=model_name)
    preds = models.ultralytics.yolov5.predict(model, valid_ds, detection_threshold=0.0)
    _test_preds(preds)


@pytest.mark.parametrize(
    "model_name",
    ["yolov5s", "yolov5m", "yolov5l"],
)
def test_yolo_predict_dl(fridge_ds, model_name):
    _, valid_ds = fridge_ds
    infer_dl = models.ultralytics.yolov5.infer_dl(valid_ds, batch_size=1, shuffle=False)
    model = models.ultralytics.yolov5.model(5, img_size=384, model_name=model_name)
    preds = models.ultralytics.yolov5.predict_dl(
        model, infer_dl, detection_threshold=0.0
    )
    _test_preds(preds)
