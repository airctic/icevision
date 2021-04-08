import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name",
    ["yolov5s", "yolov5m", "yolov5l"],
)
def test_fastai_yolo_train(fridge_ds, model_name):
    train_ds, valid_ds = fridge_ds
    train_dl = models.ultralytics.yolov5.train_dl(
        train_ds, batch_size=3, num_workers=0, shuffle=False
    )
    valid_dl = models.ultralytics.yolov5.valid_dl(
        valid_ds, batch_size=3, num_workers=0, shuffle=False
    )
    model = models.ultralytics.yolov5.model(5, img_size=384, model_name=model_name)

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    class LightModel(models.ultralytics.yolov5.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    light_model = LightModel(model, metrics=metrics)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(light_model, train_dl, valid_dl)
