import pytest
from icevision.all import *
from icevision.models.torchvision import keypoint_rcnn


@pytest.fixture
def light_model_cls():
    class LightModel(keypoint_rcnn.lightning.ModelAdapter):
        def __init__(self, model, metrics=None):
            super(LightModel, self).__init__(model, metrics)
            self.was_finalize_metrics_called = False

        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-3)

        def finalize_metrics(self):
            self.was_finalize_metrics_called = True

    return LightModel


def test_lightining_keypoints_rcnn_validate(ochuman_keypoints_dls, light_model_cls):
    _, valid_dl = ochuman_keypoints_dls
    model = keypoint_rcnn.model(num_keypoints=19)
    light_model = light_model_cls(model)
    trainer = pl.Trainer(
        max_epochs=1,
        weights_summary=None,
        num_sanity_val_steps=0,
        logger=False,
        checkpoint_callback=False,
    )

    trainer.validate(light_model, valid_dl)


def test_lightining_keypoints_finalizes_metrics_on_validation_epoch_end(
    light_model_cls,
):
    model = keypoint_rcnn.model(num_keypoints=19)
    light_model = light_model_cls(model)

    light_model.validation_epoch_end(None)

    assert light_model.was_finalize_metrics_called == True
