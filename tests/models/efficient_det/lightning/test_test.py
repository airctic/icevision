import pytest
from icevision.all import *


@pytest.fixture
def light_model_cls():
    class LightModel(models.ross.efficientdet.lightning.ModelAdapter):
        def __init__(self, model, metrics):
            super(LightModel, self).__init__(model, metrics)
            self.was_finalize_metrics_called = False

        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-3)

        def finalize_metrics(self):
            self.was_finalize_metrics_called = True

    return LightModel


# WARNING: Only works with cuda: https://github.com/rwightman/efficientdet-pytorch/issues/44#issuecomment-662594014
@pytest.mark.cuda
@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_efficientdet_test(
    fridge_efficientdet_dls, fridge_efficientdet_model, light_model_cls, metrics
):
    _, valid_dl = fridge_efficientdet_dls
    light_model = light_model_cls(fridge_efficientdet_model, metrics=metrics)
    trainer = pl.Trainer(
        max_epochs=1,
        weights_summary=None,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False,
    )

    trainer.test(light_model, valid_dl)


# WARNING: Only works with cuda: https://github.com/rwightman/efficientdet-pytorch/issues/44#issuecomment-662594014
@pytest.mark.cuda
@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_efficientdet_finalizes_metrics_on_test_epoch_end(
    fridge_efficientdet_model, light_model_cls, metrics
):
    light_model = light_model_cls(fridge_efficientdet_model, metrics=metrics)

    light_model.test_epoch_end(None)

    assert light_model.was_finalize_metrics_called == True
