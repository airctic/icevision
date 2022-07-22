import pytest
from icevision.all import *
from icevision.models.torchvision import faster_rcnn


@pytest.fixture
def light_model_cls():
    class LightModel(faster_rcnn.lightning.ModelAdapter):
        def __init__(self, model, metrics=None):
            super(LightModel, self).__init__(model, metrics)
            self.was_finalize_metrics_called = False
            self.logs = {}

        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-3)

        def finalize_metrics(self):
            self.was_finalize_metrics_called = True

        def log(self, key, value, **args):
            super(LightModel, self).log(key, value, **args)
            self.logs[key] = value

    return LightModel


@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_faster_rcnn_test(
    fridge_faster_rcnn_dls, fridge_faster_rcnn_model, light_model_cls, metrics
):
    _, valid_dl = fridge_faster_rcnn_dls
    light_model = light_model_cls(fridge_faster_rcnn_model, metrics=metrics)
    trainer = pl.Trainer(
        max_epochs=1,
        weights_summary=None,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False,
    )

    trainer.test(light_model, valid_dl)


@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_faster_rcnn_finalizes_metrics_on_test_epoch_end(
    fridge_faster_rcnn_model, light_model_cls, metrics
):
    light_model = light_model_cls(fridge_faster_rcnn_model, metrics=metrics)

    light_model.test_epoch_end(None)

    assert light_model.was_finalize_metrics_called == True


def test_lightining_faster_rcnn_logs_losses_during_test_step(
    fridge_faster_rcnn_dls, fridge_faster_rcnn_model, light_model_cls
):
    train_dl, _ = fridge_faster_rcnn_dls
    light_model = light_model_cls(fridge_faster_rcnn_model, metrics=None)
    for batch in train_dl:
        break
    light_model.convert_raw_predictions = lambda **args: None
    light_model.accumulate_metrics = lambda **args: None

    light_model.test_step(batch, 0)

    assert list(light_model.logs.keys()) == ["test_loss"]
