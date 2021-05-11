import pytest
from icevision.all import *
from icevision.models.ultralytics.yolov5.backbones import *


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_lightning_yolo_train(fridge_ds, backbone):
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

    class LightModel(models.ultralytics.yolov5.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    light_model = LightModel(model, metrics=metrics)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(light_model, train_dl, valid_dl)
