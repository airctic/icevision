import pytest
from icevision.all import *


@pytest.fixture(scope="session")
def fake_faster_rcnn_model():
    class FakeFasterRCNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            # hack for the function `model_device` to work
            self.layer = nn.Linear(1, 1)

        def forward(self, *args, **kwargs):
            return [
                {
                    "scores": tensor([0.8, 0.9]),
                    "labels": tensor([1, 2]),
                    "boxes": tensor([[10, 10, 50, 50], [20, 20, 40, 40]]),
                }
            ]

    return FakeFasterRCNNModel()


def test_faster_rcnn_show_results(fake_faster_rcnn_model, fridge_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    train_ds, valid_ds = fridge_ds

    faster_rcnn.show_results(
        model=fake_faster_rcnn_model, dataset=valid_ds, num_samples=1, ncols=1
    )


def test_get_preds(fridge_faster_rcnn_model, fridge_ds):
    model = fridge_faster_rcnn_model
    ds, _ = fridge_ds

    s, p = faster_rcnn.get_preds(model, ds)

    assert len(ds) == 2
    assert len(s) == 2
    assert len(p) == 2


def test_plot_losses(fridge_faster_rcnn_model, fridge_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    model = fridge_faster_rcnn_model
    ds, _ = fridge_ds

    samples_plus_losses, preds, _ = faster_rcnn.plot_top_losses(
        model=model, dataset=ds, sort_by="loss_total", n_samples=2
    )
    assert len(samples_plus_losses) == len(ds) == len(preds)


def test_get_losses(fridge_faster_rcnn_model, fridge_ds):
    model = fridge_faster_rcnn_model
    ds, _ = fridge_ds

    samples, losses_stats = faster_rcnn.get_losses(model, ds)
    assert len(samples) == len(ds)
    assert set(losses_stats.keys()) == {
        "loss_box_reg",
        "loss_classifier",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_total",
    }
    assert "loss_box_reg" in samples[0].keys()
    assert "loss_classifier" in samples[0]["text"]
    assert "IMG" in samples[0]["text"]
