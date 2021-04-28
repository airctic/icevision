import pytest
from icevision.all import *


def _test_dl(x, y, recs):
    assert len(recs) == 3
    assert recs[0].img is None
    # assert recs[0].img.shape == recs[1].img.shape == recs[2].img.shape == (384, 384, 3)

    assert x.shape == torch.Size([3, 3, 384, 384])
    if y is not None:
        assert y.shape[1] == 6

        # check batch size of 3 inside y
        assert np.all(np.unique(y[:, 0].numpy().astype(int)) == np.array([0, 1, 2]))

        labels_in_batch = np.unique(y[:, 1].numpy().astype(int))
        possible_labels = np.arange(
            0, 4 + 1
        )  # 4 = number of classes in parser.class_map - 1 (given we don't have background)
        # is labels_in_batch a subset of possible_labels?
        assert len(np.setdiff1d(labels_in_batch, possible_labels)) == 0


def test_train_dataloader(fridge_ds):
    train_ds, _ = fridge_ds
    train_dl = models.ultralytics.yolov5.train_dl(
        train_ds, batch_size=3, num_workers=0, shuffle=False
    )
    (x, y), records = first(train_dl)

    _test_dl(x, y, records)


def test_val_dataloader(fridge_ds):
    _, valid_ds = fridge_ds
    valid_dl = models.ultralytics.yolov5.valid_dl(
        valid_ds, batch_size=3, num_workers=0, shuffle=False
    )
    (x, y), records = first(valid_dl)

    _test_dl(x, y, records)


def test_infer_dataloader(fridge_ds):
    _, valid_ds = fridge_ds
    infer_dl = models.ultralytics.yolov5.infer_dl(
        valid_ds, batch_size=3, num_workers=0, shuffle=False
    )
    (x,), records = first(infer_dl)

    _test_dl(x, None, records)
