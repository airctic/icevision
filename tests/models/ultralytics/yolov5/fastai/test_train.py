import pytest
from icevision.all import *
from icevision.models.ultralytics.yolov5.backbones import *


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_fastai_yolo_train(fridge_ds, backbone):
    train_ds, valid_ds = fridge_ds
    train_dl = models.ultralytics.yolov5.train_dl(
        train_ds, batch_size=3, num_workers=0, shuffle=False
    )
    valid_dl = models.ultralytics.yolov5.valid_dl(
        valid_ds, batch_size=3, num_workers=0, shuffle=False
    )
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=384, backbone=backbone(pretrained=True)
    )

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    learn = models.ultralytics.yolov5.fastai.learner(
        dls=[train_dl, valid_dl], model=model, metrics=metrics
    )

    learn.fine_tune(1, 1e-4)
