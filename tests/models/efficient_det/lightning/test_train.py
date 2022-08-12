import pytest
import random
from icevision.all import *


@pytest.fixture
def light_model_cls():
    class LightModel(models.ross.efficientdet.lightning.ModelAdapter):
        def __init__(self, model, metrics):
            super(LightModel, self).__init__(model, metrics)
            self.model = model
            self.logs = {}

        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-3)

        def log(self, key, value, **args):
            super(LightModel, self).log(key, value, **args)
            self.logs[key] = value

    return LightModel


# WARNING: Only works with cuda: https://github.com/rwightman/efficientdet-pytorch/issues/44#issuecomment-662594014
@pytest.mark.cuda
@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_efficientdet_train(
    fridge_efficientdet_dls, fridge_efficientdet_model, light_model_cls, metrics
):
    train_dl, valid_dl = fridge_efficientdet_dls
    light_model = light_model_cls(model=fridge_efficientdet_model, metrics=metrics)
    trainer = pl.Trainer(
        max_epochs=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(light_model, train_dl, valid_dl)


def test_lightining_efficientdet_training_step_returns_loss(
    fridge_efficientdet_dls, fridge_efficientdet_model, light_model_cls
):
    with torch.set_grad_enabled(True):
        train_dl, _ = fridge_efficientdet_dls
        light_model = light_model_cls(model=fridge_efficientdet_model, metrics=None)
        for batch in train_dl:
            break
        expected_loss = random.randint(0, 1000)

        def fake_compute_loss(self, *args):
            return expected_loss

        light_model.compute_loss = fake_compute_loss

        loss = light_model.training_step(batch, 0)

        assert loss == expected_loss


def test_lightining_efficientdet_logs_losses_during_training_step(
    fridge_efficientdet_dls, fridge_efficientdet_model, light_model_cls
):
    with torch.set_grad_enabled(True):
        train_dl, _ = fridge_efficientdet_dls
        light_model = light_model_cls(model=fridge_efficientdet_model, metrics=None)
        for batch in train_dl:
            break

        light_model.training_step(batch, 0)

        assert sorted(light_model.logs.keys()) == sorted(
            ["train_loss", "train_box_loss", "train_class_loss"]
        )
