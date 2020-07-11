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


@pytest.fixture()
def records(img, labels, bboxes):
    return [{"img": img, "labels": labels, "bboxes": bboxes}] * 2


# labels = [1, 2]
# bboxes = [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(4, 3, 2, 1)]
# img = 255 * np.ones((4, 4, 3), dtype=np.uint8)
# records = [{"img": img, "labels": labels, "bboxes": bboxes}] * 2


def _test_batch(batch):
    images, targets = batch

    assert isinstance(images, list)
    assert len(images) == 2
    assert images[0].dtype == torch.float
    assert images[0].shape == (3, 4, 4)

    assert isinstance(targets, list)
    assert isinstance(targets[0], dict)
    assert len(targets) == 2

    for target in targets:
        assert set(target) == {"labels", "boxes"}

        assert target["labels"].dtype == torch.int64
        assert torch.all(target["labels"] == tensor([1, 2], dtype=torch.int64))

        assert target["boxes"].dtype == torch.float
        assert torch.all(
            target["boxes"] == tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.float)
        )


def test_faster_rcnn_build_train_batch(records):
    batch = faster_rcnn.build_train_batch(records)
    _test_batch(batch=batch)


def test_faster_rcnn_build_valid_batch(records):
    batch = faster_rcnn.build_valid_batch(records)
    _test_batch(batch=batch)
