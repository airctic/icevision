from mantisshrimp import *
from mantisshrimp.imports import first, torch, tensor, Tensor

_fake_box = [0, 1, 2, 3]


def test_build_training_sample_maskrcnn(data_sample):
    x, y = MantisMaskRCNN.build_training_sample(**data_sample)
    assert x.dtype == torch.float32
    assert x.shape == (3, 427, 640)
    assert isinstance(y, dict)
    assert set(y.keys()) == {"image_id", "labels", "boxes", "masks"}
    assert y["image_id"].dtype == torch.int64
    assert y["labels"].dtype == torch.int64
    assert y["labels"].shape == (16,)
    assert (
        y["labels"] == tensor([6, 1, 1, 1, 1, 1, 1, 1, 31, 31, 1, 3, 31, 1, 31, 31])
    ).all()
    assert y["boxes"].dtype == torch.float32
    assert y["boxes"].dtype == torch.float32
    assert y["boxes"].shape == (16, 4)
    assert y["masks"].dtype == torch.uint8
    assert y["masks"].shape == (16, 427, 640)


def test_build_training_sample_fasterrcnn(data_sample):
    x, y = MantisFasterRCNN.build_training_sample(**data_sample)
    assert x.dtype == torch.float32
    assert x.shape == (3, 427, 640)
    assert isinstance(y, dict)
    assert set(y.keys()) == {"image_id", "labels", "boxes"}
    assert y["image_id"].dtype == torch.int64
    assert y["labels"].dtype == torch.int64
    assert y["labels"].shape == (16,)
    assert (
        y["labels"] == tensor([6, 1, 1, 1, 1, 1, 1, 1, 31, 31, 1, 3, 31, 1, 31, 31])
    ).all()
    assert y["boxes"].dtype == torch.float32
    assert y["boxes"].shape == (16, 4)


def test_rcnn_empty_training_sample(data_sample):
    data_sample["bbox"] = []
    data_sample["label"] = []
    x, y = MantisMaskRCNN.build_training_sample(**data_sample)
    assert (y["boxes"] == tensor([_fake_box], dtype=torch.float)).all()
    assert y["labels"] == tensor([0])
