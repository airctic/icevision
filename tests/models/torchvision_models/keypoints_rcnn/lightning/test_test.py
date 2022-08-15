import pytest
from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


@pytest.fixture
def light_model_cls():
    class LightModel(keypoint_rcnn.lightning.ModelAdapter):
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


def test_lightining_keypoints_rcnn_test(ochuman_keypoints_dls, light_model_cls):
    _, valid_dl = ochuman_keypoints_dls
    model = keypoint_rcnn.model(num_keypoints=19)
    light_model = light_model_cls(model)
    trainer = pl.Trainer(
        max_epochs=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.test(light_model, valid_dl)


def test_lightining_keypoints_finalizes_metrics_on_test_epoch_end(light_model_cls):
    with torch.set_grad_enabled(False):
        model = keypoint_rcnn.model(num_keypoints=19)
        light_model = light_model_cls(model)

        light_model.test_epoch_end(None)

        assert light_model.was_finalize_metrics_called == True


def test_lightining_keypoints_rcnn_logs_losses_during_test_step(
    ochuman_keypoints_dls, light_model_cls
):
    with torch.set_grad_enabled(False):
        _, valid_dl = ochuman_keypoints_dls
        model = keypoint_rcnn.model(num_keypoints=19)
        light_model = light_model_cls(model)
        for batch in valid_dl:
            break
        light_model.convert_raw_predictions = lambda **args: None
        light_model.accumulate_metrics = lambda **args: None

        light_model.test_step(batch, 0)

        assert list(light_model.logs.keys()) == ["test_loss"]
