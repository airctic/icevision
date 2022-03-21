import pytest
from icevision.all import *
from icevision.models.mmseg.models.deeplabv3plus.backbones import *
from icevision.models.mmseg.models import *


@pytest.mark.parametrize("backbone", [resnet18_d8])
def test_pl_train(camvid_ds, backbone):
    train_ds, valid_ds = camvid_ds
    train_dl = deeplabv3plus.train_dl(
        train_ds, batch_size=4, num_workers=0, shuffle=False
    )
    valid_dl = deeplabv3plus.valid_dl(
        valid_ds, batch_size=1, num_workers=0, shuffle=False
    )
    model = deeplabv3plus.model(num_classes=32, backbone=backbone(pretrained=False))

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
