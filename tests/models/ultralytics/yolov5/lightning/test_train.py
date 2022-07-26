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

    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=1, gpus=gpus)
    trainer.fit(light_model, train_dl, valid_dl)


@pytest.mark.parametrize(
    "backbone",
    [small, medium],
)
def test_lightning_yolo_training_step_returns_loss(fridge_ds, backbone):
    with torch.set_grad_enabled(True):
        train_ds, _ = fridge_ds
        train_dl = models.ultralytics.yolov5.train_dl(
            train_ds, batch_size=1, num_workers=0, shuffle=False
        )
        model = models.ultralytics.yolov5.model(
            num_classes=5, img_size=384, backbone=backbone(pretrained=False)
        )

        class LightModel(models.ultralytics.yolov5.lightning.ModelAdapter):
            def configure_optimizers(self):
                return SGD(self.parameters(), lr=1e-4)

        light_model = LightModel(model)
        light_model.to("cpu")
        expected_loss = random.randint(0, 10)
        light_model.compute_loss = lambda *args: [expected_loss]
        for batch in train_dl:
            batch
            break

        loss = light_model.training_step(batch, 0)

        assert loss == expected_loss


@pytest.mark.parametrize(
    "backbone",
    [small, medium],
)
def test_lightning_yolo_logs_losses_during_training_step(fridge_ds, backbone):
    with torch.set_grad_enabled(True):
        train_ds, _ = fridge_ds
        train_dl = models.ultralytics.yolov5.train_dl(
            train_ds, batch_size=1, num_workers=0, shuffle=False
        )
        model = models.ultralytics.yolov5.model(
            num_classes=5, img_size=384, backbone=backbone(pretrained=False)
        )

        class LightModel(models.ultralytics.yolov5.lightning.ModelAdapter):
            def __init__(self, model, metrics=None):
                super(LightModel, self).__init__(model, metrics)
                self.model = model
                self.logs = {}

            def configure_optimizers(self):
                return SGD(self.parameters(), lr=1e-4)

            def log(self, key, value, **args):
                super(LightModel, self).log(key, value, **args)
                self.logs[key] = value

        light_model = LightModel(model)
        light_model.to("cpu")
        light_model.compute_loss = lambda *args: [random.randint(0, 10)]
        for batch in train_dl:
            batch
            break

        light_model.training_step(batch, 0)

        assert list(light_model.logs.keys()) == [f"train_loss"]
