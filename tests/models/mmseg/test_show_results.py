import pytest
from icevision.all import *
from icevision.models.fastai.unet.backbones import *
from icevision.models.interpretation import get_samples_losses


# @pytest.mark.parametrize(
#     "backbone",
#     [resnet18, resnet50, resnet101],
# )
# def test_show_results(camvid_ds, backbone, monkeypatch):
#     monkeypatch.setattr(plt, "show", lambda: None)
#     _, valid_ds = camvid_ds
#     model = models.fastai.unet.model(
#         num_classes=32, img_size=64, backbone=backbone(pretrained=True)
#     )
#     models.fastai.unet.show_results(model, valid_ds)


# @pytest.mark.parametrize(
#     "backbone",
#     [resnet18, resnet50, resnet101],
# )
def test_plot_losses(camvid_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ds = camvid_ds

    backbone = models.mmseg.deeplabv3.backbones.resnet50_d8

    model = models.mmseg.deeplabv3.model(
        backbone=backbone(
            pretrained=False,
        ),
        num_classes=32,
    )

    model.eval()  # Needed to allow batch size of 1
    model.cuda()  # Needed when ran on machine with a GPU

    samples_plus_losses, preds, _ = models.mmseg.deeplabv3.interp.plot_top_losses(
        model=model, dataset=ds, sort_by="loss_total", n_samples=2
    )
    assert len(samples_plus_losses) == len(ds) == len(preds)

    loss_per_image = get_samples_losses(samples_plus_losses)
    assert "filepath" in loss_per_image[0].keys()
