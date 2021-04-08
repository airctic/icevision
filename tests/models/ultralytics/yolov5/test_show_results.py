import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "model_name",
    ["yolov5s", "yolov5m", "yolov5l"],
)
def test_show_results(fridge_ds, model_name, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, valid_ds = fridge_ds
    model = models.ultralytics.yolov5.model(5, img_size=384, model_name=model_name)
    models.ultralytics.yolov5.show_results(model, valid_ds)


@pytest.mark.parametrize(
    "model_name",
    ["yolov5s", "yolov5m", "yolov5l"],
)
def test_plot_losses(fridge_ds, model_name, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ds = fridge_ds
    model = models.ultralytics.yolov5.model(5, img_size=384, model_name=model_name)

    samples_plus_losses, preds, _ = models.ultralytics.yolov5.interp.plot_top_losses(
        model=model, dataset=ds, sort_by="loss_yolo", n_samples=2
    )
    assert len(samples_plus_losses) == len(ds) == len(preds)
