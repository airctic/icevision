import pytest
from icevision.all import *
from icevision.models.ultralytics.yolov5.backbones import *


@pytest.fixture
def light_model_cls():
    class LightModel(models.ultralytics.yolov5.lightning.ModelAdapter):
        def __init__(self, model, metrics=None):
            super(LightModel, self).__init__(model, metrics)
            self.was_finalize_metrics_called = False

        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-3)

        def finalize_metrics(self):
            self.was_finalize_metrics_called = True

    return LightModel


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_lightning_yolo_test(fridge_ds, backbone, light_model_cls):
    _, valid_ds = fridge_ds
    valid_dl = models.ultralytics.yolov5.valid_dl(
        valid_ds, batch_size=3, num_workers=0, shuffle=False
    )
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=384, backbone=backbone(pretrained=True)
    )
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    light_model = light_model_cls(model, metrics=metrics)
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=1, gpus=gpus)

    trainer.test(light_model, valid_dl)


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_lightning_yolo_finalizes_metrics_on_test_epoch_end(backbone, light_model_cls):
    model = models.ultralytics.yolov5.model(
        num_classes=5, img_size=384, backbone=backbone(pretrained=True)
    )
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    light_model = light_model_cls(model, metrics=metrics)

    light_model.test_epoch_end(None)

    assert light_model.was_finalize_metrics_called == True
