import pytest
from icevision.all import *


@pytest.fixture
def light_model_cls():
    class LightModel(efficientdet.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-3)

    return LightModel


# WARNING: Only works with cuda: https://github.com/rwightman/efficientdet-pytorch/issues/44#issuecomment-662594014
@pytest.mark.cuda
@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_efficientdet_train(
    fridge_efficientdet_dls, fridge_efficientdet_model, light_model_cls, metrics
):
    train_dl, valid_dl = fridge_efficientdet_dls
    light_model = light_model_cls(fridge_efficientdet_model, metrics=metrics)

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=1,
        weights_summary=None,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)
