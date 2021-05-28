import pytest
from icevision.all import *
from icevision.models.fastai.unet.backbones import *


@pytest.mark.parametrize(
    "backbone",
    [resnet18, resnet50, resnet101],
)
def test_show_results(camvid_ds, backbone, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, valid_ds = camvid_ds
    model = models.fastai.unet.model(
        num_classes=32, img_size=64, backbone=backbone(pretrained=True)
    )
    models.fastai.unet.show_results(model, valid_ds)
