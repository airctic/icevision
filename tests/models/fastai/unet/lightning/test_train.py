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
        weights_summary=None,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)
