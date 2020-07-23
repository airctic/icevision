import pytest
from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.models.rcnn import faster_rcnn, mask_rcnn


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


### Faster RCNN ###
def _test_common_rcnn_batch(batch):
    (images, targets), records = batch

    assert isinstance(images, list)
    assert len(images) == 2
    assert images[0].dtype == torch.float
    assert images[0].shape == (3, 4, 4)

    assert isinstance(targets, list)
    assert isinstance(targets[0], dict)
    assert len(targets) == 2

    for target in targets:
        assert target["labels"].dtype == torch.int64
        assert torch.all(target["labels"] == tensor([1, 2], dtype=torch.int64))

        assert target["boxes"].dtype == torch.float
        assert torch.all(
            target["boxes"] == tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.float)
        )


def _test_faster_rcnn_batch(batch):
    _test_common_rcnn_batch(batch)

    (images, targets), records = batch
    for target in targets:
        assert set(target) == {"labels", "boxes"}


def _test_infer_batch(batch):
    assert len(batch) == 2

    for x in batch:
        assert x.shape == (3, 4, 4)
        assert x.dtype == torch.float


def test_faster_rcnn_build_train_batch(records):
    batch = faster_rcnn.build_train_batch(records)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_build_valid_batch(records):
    batch = faster_rcnn.build_valid_batch(records)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_build_infer_batch(img):
    images = [img] * 2
    batch = faster_rcnn.build_infer_batch(images=images)
    _test_infer_batch(batch)


def test_faster_rcnn_train_dataloader(records):
    dl = faster_rcnn.train_dataloader(records, batch_size=2)
    batch = first(dl)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_valid_dataloader(records):
    dl = faster_rcnn.valid_dataloader(records, batch_size=2)
    batch = first(dl)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_infer_dataloader(img):
    images = [img] * 2
    dl = faster_rcnn.infer_dataloader(dataset=images, batch_size=2)
    batch = first(dl)
    _test_infer_batch(batch=batch)


### Mask RCNN ###
@pytest.fixture()
def masks():
    return MaskArray(np.ones((1, 4, 4), dtype=np.uint8))


@pytest.fixture()
def mask_records(img, labels, bboxes, masks):
    return [{"img": img, "labels": labels, "bboxes": bboxes, "masks": masks}] * 2


def _test_mask_rcnn_batch(batch):
    _test_common_rcnn_batch(batch=batch)
    (images, targets), records = batch

    for target in targets:
        assert set(target.keys()) == {"labels", "boxes", "masks"}
        assert target["masks"].shape == (1, 4, 4)
        assert target["masks"].dtype == torch.uint8


def test_mask_rcnn_build_train_batch(mask_records):
    batch = mask_rcnn.build_train_batch(mask_records)
    _test_mask_rcnn_batch(batch)


def test_mask_rcnn_build_valid_batch(mask_records):
    batch = mask_rcnn.build_valid_batch(mask_records)
    _test_mask_rcnn_batch(batch)


def test_mask_rcnn_build_infer_batch(img):
    test_faster_rcnn_infer_dataloader(img)


def test_mask_rcnn_train_dataloader(mask_records):
    dl = mask_rcnn.train_dataloader(mask_records, batch_size=2)
    batch = first(dl)
    _test_mask_rcnn_batch(batch=batch)


def test_mask_rcnn_valid_dataloader(mask_records):
    dl = mask_rcnn.valid_dataloader(mask_records, batch_size=2)
    batch = first(dl)
    _test_mask_rcnn_batch(batch=batch)


def test_mask_rcnn_infer_dataloader(img):
    test_faster_rcnn_infer_dataloader(img)
