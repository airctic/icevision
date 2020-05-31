from mantisshrimp import *
from mantisshrimp.imports import first, torch, tensor, Tensor
from mantisshrimp.data.rcnn_dataloader import _fake_box


def test_item2rcnn_training_sample(item):
    x, y = item2rcnn_training_sample(item)
    assert x.dtype == torch.float32
    assert x.shape == (3, 427, 640)
    assert isinstance(y, dict)
    assert list(y.keys()) == ["image_id", "labels", "iscrowd", "boxes", "area", "masks"]
    assert y["image_id"].dtype == torch.int64
    assert y["labels"].dtype == torch.int64
    assert y["labels"].shape == (16,)
    assert (
        y["labels"] == tensor([6, 1, 1, 1, 1, 1, 1, 1, 27, 27, 1, 3, 27, 1, 27, 27])
    ).all()
    assert y["iscrowd"].dtype == torch.uint8
    assert y["iscrowd"].shape == (16,)
    assert (
        y["iscrowd"] == tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ).all()
    assert y["boxes"].dtype == torch.float32
    assert y["boxes"].shape == (16, 4)
    assert y["area"].shape == (16,)
    assert y["masks"].dtype == torch.uint8
    assert y["masks"].shape == (16, 427, 640)


def test_item2rcnn_training_sample_empty(item):
    item = item.replace(bboxes=[], labels=[], iscrowds=[])
    x, y = item2rcnn_training_sample(item)
    assert (y["boxes"] == tensor([_fake_box], dtype=torch.float)).all()
    assert y["labels"] == tensor([0])
    assert y["iscrowd"] == tensor([0])
    assert y["area"] == tensor([4])


def test_rcnn_dataloader():
    train_dl, valid_dl = test_utils.sample_rcnn_dataloaders()
    xb, yb = first(train_dl)
    assert len(xb) == 2
    assert len(yb) == 2

    x = xb[0]
    assert x.shape == (3, 427, 640)
    assert isinstance(x, Tensor)

    y = yb[0]
    assert y["image_id"] == tensor(0)
    assert allequal([isinstance(o, Tensor) for o in y.values()])
    assert (
        y["labels"] == tensor([6, 1, 1, 1, 1, 1, 1, 1, 27, 27, 1, 3, 27, 1, 27, 27])
    ).all()
    assert (
        y["iscrowd"] == tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ).all()
    assert (y["boxes"][0] == tensor([0.0000, 73.8900, 416.4400, 379.0200])).all()
    assert y["masks"].shape == (16, 427, 640)
    assert not (y["masks"] == 0).all()
    assert allequal(lmap(len, [y["labels"], y["iscrowd"], y["boxes"], y["masks"]]))
