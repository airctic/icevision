import pytest
from icevision.all import *
from icevision.models.ultralytics.yolov5.backbones import *
import albumentations as A


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_e2e_detect(samples_source, fridge_class_map, backbone):
    img_path = samples_source / "fridge/odFridgeObjects/images/10.jpg"
    tfms_ = tfms.A.Adapter([A.Resize(384, 384), A.Normalize()])
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=384, backbone=backbone(pretrained=True)
    )
    pred_dict = models.ultralytics.yolov5.end2end_detect(
        img_path, tfms_, model, fridge_class_map
    )
    assert len(pred_dict["detection"]["bboxes"]) == 0


def _test_preds(preds):
    assert len(preds) == 3
    assert isinstance(preds[0].detection.bboxes[0], BBox)
    assert len(preds[0].detection.scores) == len(preds[0].detection.labels)
    assert len(preds[2].detection.scores) == len(preds[2].detection.labels)


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_yolo_predict(fridge_ds, backbone):
    _, valid_ds = fridge_ds
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=384, backbone=backbone(pretrained=True)
    )
    preds = models.ultralytics.yolov5.predict(model, valid_ds, detection_threshold=0.0)
    _test_preds(preds)


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_yolo_predict_from_dl(fridge_ds, backbone):
    _, valid_ds = fridge_ds
    infer_dl = models.ultralytics.yolov5.infer_dl(valid_ds, batch_size=1, shuffle=False)
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=384, backbone=backbone(pretrained=True)
    )
    preds = models.ultralytics.yolov5.predict_from_dl(
        model, infer_dl, detection_threshold=0.0
    )
    _test_preds(preds)
