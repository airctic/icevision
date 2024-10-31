import pytest
from icevision.all import *

from icevision.models.mmseg.models.deeplabv3.backbones import *
from icevision.models.mmseg.models import *
from icevision.models.interpretation import get_samples_losses


@pytest.mark.parametrize(
    "backbone",
    [resnet18_d8, resnet50_d8, resnet101_d8],
)
def test_show_results(camvid_ds, backbone, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, valid_ds = camvid_ds
    model = deeplabv3.model(
        backbone=backbone(
            pretrained=False,
        ),
        num_classes=32,
    )

    model.eval()  # Needed to allow batch size of 1

    if torch.cuda.is_available():
        model.cuda()  # Needed when ran on machine with a GPU as data will be loaded on the device by default
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    deeplabv3.show_results(model, valid_ds, device=device)


@pytest.mark.parametrize(
    "backbone",
    [resnet18_d8, resnet50_d8, resnet101_d8],
)
def test_plot_losses(camvid_ds, backbone, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ds = camvid_ds

    model = deeplabv3.model(
        backbone=backbone(
            pretrained=False,
        ),
        num_classes=32,
    )

    model.eval()  # Needed to allow batch size of 1

    if torch.cuda.is_available():
        model.cuda()  # Needed when ran on machine with a GPU as data will be loaded on the device by default
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    samples_plus_losses, preds, _ = deeplabv3.interp.plot_top_losses(
        model=model, dataset=ds, sort_by="loss_total", n_samples=2, device=device
    )
    assert len(samples_plus_losses) == len(ds) == len(preds)

    loss_per_image = get_samples_losses(samples_plus_losses)
    assert "filepath" in loss_per_image[0].keys()
