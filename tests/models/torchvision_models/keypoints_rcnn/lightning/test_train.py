import pytest
from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


@pytest.fixture
def light_model_cls():
    class LightModel(keypoint_rcnn.lightning.ModelAdapter):
        def __init__(self, model, metrics=None):
            super(LightModel, self).__init__(model, metrics)
            self.model = model
            self.logs = {}

        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

        def log(self, key, value, **args):
            super(LightModel, self).log(key, value, **args)
            self.logs[key] = value

    return LightModel


def test_lightining_keypoints_rcnn_train(ochuman_keypoints_dls, light_model_cls):
    train_dl, valid_dl = ochuman_keypoints_dls
    model = keypoint_rcnn.model(num_keypoints=19)
    light_model = light_model_cls(model)

    trainer = pl.Trainer(
        max_epochs=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)


def test_lightining_keypoints_rcnn_training_step_returns_loss(
    ochuman_keypoints_dls, light_model_cls
):
    with torch.set_grad_enabled(True):
        train_dl, _ = ochuman_keypoints_dls
        model = keypoint_rcnn.model(num_keypoints=19)
        light_model = light_model_cls(model)
        expected_loss = random.randint(0, 10)
        light_model.compute_loss = lambda *args: expected_loss
        for batch in train_dl:
            break

        loss = light_model.training_step(batch, 0)

        assert loss == expected_loss


def test_lightining_keypoints_rcnn_logs_losses_during_training_step(
    ochuman_keypoints_dls, light_model_cls
):
    with torch.set_grad_enabled(True):
        train_dl, _ = ochuman_keypoints_dls
        model = keypoint_rcnn.model(num_keypoints=19)
        light_model = light_model_cls(model)
        for batch in train_dl:
            break

        light_model.training_step(batch, 0)

        assert list(light_model.logs.keys()) == ["train_loss"]
