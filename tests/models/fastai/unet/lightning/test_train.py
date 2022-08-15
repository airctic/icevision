import pytest
from icevision.all import *
from icevision.models.fastai.unet.backbones import *


@pytest.mark.parametrize("backbone", [resnet18])
def test_fastai_unet_train(camvid_ds, backbone):
    train_ds, valid_ds = camvid_ds
    train_dl = models.fastai.unet.train_dl(
        train_ds, batch_size=4, num_workers=0, shuffle=False
    )
    valid_dl = models.fastai.unet.valid_dl(
        valid_ds, batch_size=1, num_workers=0, shuffle=False
    )
    model = models.fastai.unet.model(
        num_classes=32, img_size=64, backbone=backbone(pretrained=False)
    )

    class LightModel(models.fastai.unet.lightning.ModelAdapter):
        def configure_optimizers(self):
            return Adam(self.parameters(), lr=1e-4)

    light_model = LightModel(model)

    trainer = pl.Trainer(
        max_epochs=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)


@pytest.mark.parametrize("backbone", [resnet18])
def test_fastai_unet_training_step_returns_loss(camvid_ds, backbone):
    with torch.set_grad_enabled(True):
        train_ds, _ = camvid_ds
        train_dl = models.fastai.unet.train_dl(
            train_ds, batch_size=1, num_workers=0, shuffle=False
        )
        model = models.fastai.unet.model(
            num_classes=32, img_size=64, backbone=backbone(pretrained=False)
        )

        class LightModel(models.fastai.unet.lightning.ModelAdapter):
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-4)

        for batch in train_dl:
            break
        expected_loss = random.randint(0, 1000)
        light_model = LightModel(model)
        light_model.compute_loss = lambda *args: expected_loss

        loss = light_model.training_step(batch, 0)

        assert loss == expected_loss


@pytest.mark.parametrize("backbone", [resnet18])
def test_fastai_unet_logs_losses_during_training_step(camvid_ds, backbone):
    with torch.set_grad_enabled(True):
        train_ds, _ = camvid_ds
        train_dl = models.fastai.unet.train_dl(
            train_ds, batch_size=1, num_workers=0, shuffle=False
        )
        model = models.fastai.unet.model(
            num_classes=32, img_size=64, backbone=backbone(pretrained=False)
        )

        class LightModel(models.fastai.unet.lightning.ModelAdapter):
            def __init__(self, model, metrics=None):
                super(LightModel, self).__init__(model, metrics)
                self.model = model
                self.logs = {}

            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-4)

            def log(self, key, value, **args):
                super(LightModel, self).log(key, value, **args)
                self.logs[key] = value

        for batch in train_dl:
            break
        light_model = LightModel(model)

        light_model.training_step(batch, 0)

        assert list(light_model.logs.keys()) == ["train_loss"]
