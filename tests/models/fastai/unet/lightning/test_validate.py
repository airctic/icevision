import pytest
from icevision.all import *
from icevision.models.fastai.unet.backbones import *


@pytest.mark.parametrize("backbone", [resnet18])
def test_fastai_unet_validate(camvid_ds, backbone):
    _, valid_ds = camvid_ds
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
    trainer.validate(light_model, valid_dl)


@pytest.mark.parametrize("backbone", [resnet18])
def test_fastai_unet_logs_losses_during_validation_step(camvid_ds, backbone):
    with torch.set_grad_enabled(False):
        _, valid_ds = camvid_ds
        valid_dl = models.fastai.unet.train_dl(
            valid_ds, batch_size=1, num_workers=0, shuffle=False
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

        for batch in valid_dl:
            break
        light_model = LightModel(model)

        light_model.validation_step(batch, 0)

        assert list(light_model.logs.keys()) == ["val_loss"]
