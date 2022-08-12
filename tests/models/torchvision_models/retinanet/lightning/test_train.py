import pytest
from icevision.all import *
from icevision.models.torchvision import retinanet


@pytest.fixture
def light_model_cls():
    class LightModel(retinanet.lightning.ModelAdapter):
        def __init__(self, model, metrics=None):
            super(LightModel, self).__init__(model, metrics)
            self.logs = {}

        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

        def log(self, key, value, **args):
            super(LightModel, self).log(key, value, **args)
            self.logs[key] = value

    return LightModel


@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_retinanet_train(
    fridge_faster_rcnn_dls, fridge_retinanet_model, light_model_cls, metrics
):
    train_dl, valid_dl = fridge_faster_rcnn_dls
    light_model = light_model_cls(fridge_retinanet_model, metrics=metrics)

    trainer = pl.Trainer(
        max_epochs=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)


def test_lightining_retinanet_training_step_returns_loss(
    fridge_faster_rcnn_dls, fridge_retinanet_model, light_model_cls
):
    with torch.set_grad_enabled(True):
        train_dl, _ = fridge_faster_rcnn_dls
        light_model = light_model_cls(fridge_retinanet_model)
        expected_loss = random.randint(0, 10)
        light_model.compute_loss = lambda *args: expected_loss
        for batch in train_dl:
            break

        loss = light_model.training_step(batch, 0)

        assert loss == expected_loss


def test_lightining_retinanet_logs_losses_during_training_step(
    fridge_faster_rcnn_dls, fridge_retinanet_model, light_model_cls
):
    with torch.set_grad_enabled(True):
        train_dl, _ = fridge_faster_rcnn_dls
        light_model = light_model_cls(fridge_retinanet_model)
        for batch in train_dl:
            break

        light_model.training_step(batch, 0)

        assert list(light_model.logs.keys()) == ["train_loss"]
