import pytest
from mantisshrimp.imports import *
from mantisshrimp import *


@pytest.fixture()
def img():
    return 255 * np.ones((4, 4, 3), dtype=np.uint8)


@pytest.fixture()
def labels():
    return [1, 2]


@pytest.fixture()
def bboxes():
    return [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(4, 3, 2, 1)]


def test_efficient_det_build_train_sample(img, labels, bboxes):
    image, target = MantisEfficientDet.build_train_sample(
        img=img, labels=labels, bboxes=bboxes
    )

    assert torch.all(image == torch.ones((3, 4, 4), dtype=torch.float32))

    assert torch.all(target["cls"] == tensor([1, 2], dtype=torch.int64))
    assert torch.all(
        target["boxes"] == tensor([[2, 1, 4, 3], [3, 4, 1, 2]], dtype=torch.float64)
    )


def test_efficient_det_build_valid_sample(img, labels, bboxes):
    image, target = MantisEfficientDet.build_valid_sample(
        img=img, labels=labels, bboxes=bboxes
    )

    assert torch.all(image == torch.ones((3, 4, 4), dtype=torch.float32))

    assert torch.all(target["cls"] == tensor([1, 2], dtype=torch.int64))
    assert torch.all(
        target["boxes"] == tensor([[2, 1, 4, 3], [3, 4, 1, 2]], dtype=torch.float64)
    )
    assert torch.all(target["img_scale"] == tensor([1.0, 1.0]))
    assert torch.all(target["img_size"] == tensor([[4, 4], [4, 4]]))


def test_efficient_det_dataloader(img, labels, bboxes):
    dataset = [{"img": img, "labels": labels, "bboxes": bboxes}] * 2

    dl = MantisEfficientDet.dataloader(dataset, batch_size=2)
    xb, yb = first(dl)

    assert xb.shape == (2, 3, 4, 4)

    assert len(yb) == 2
    for y in yb:
        assert set(y.keys()) == {"cls", "boxes"}


def test_efficient_det_valid_dataloader(img, labels, bboxes):
    dataset = [{"img": img, "labels": labels, "bboxes": bboxes}] * 2

    dl = MantisEfficientDet.valid_dataloader(dataset, batch_size=2)
    xb, yb = first(dl)

    assert xb.shape == (2, 3, 4, 4)

    assert len(yb) == 2
    for y in yb:
        assert set(y.keys()) == {"cls", "boxes", "img_size", "img_scale"}
