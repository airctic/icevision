from icevision.all import *


def test_plot_losses(fridge_efficientdet_model, fridge_ds, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    model = fridge_efficientdet_model
    ds, _ = fridge_ds
    by = {
        "method": "weighted",
        "weights": {"class_loss": 0, "box_loss": 0},
    }

    samples_plus_losses, preds, _ = efficientdet.interp.plot_top_losses(
        model=model, dataset=ds, sort_by=by, n_samples=2
    )
    assert len(samples_plus_losses) == len(ds) == len(preds)
