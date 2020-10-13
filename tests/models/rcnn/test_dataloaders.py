import pytest
from icevision.all import *


# TODO: Duplicated fixture between here and efficientdet:test_dataloaders
@pytest.fixture()
def img():
    return 255 * np.ones((4, 4, 3), dtype=np.uint8)


@pytest.fixture()
def labels():
    return [1, 2]


@pytest.fixture()
def bboxes():
    return [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(10, 20, 30, 40)]


@pytest.fixture()
def records(img, labels, bboxes):
    Record = create_mixed_record(
        (ImageRecordMixin, LabelsRecordMixin, BBoxesRecordMixin)
    )
    record = Record()
    record.set_imageid(1)
    record.set_img(img)
    record.add_labels(labels)
    record.add_bboxes(bboxes)
    return [record] * 2


@pytest.fixture()
def dataset(img):
    return Dataset.from_images([img] * 2)


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
            target["boxes"]
            == tensor([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=torch.float)
        )


def _test_faster_rcnn_batch(batch):
    _test_common_rcnn_batch(batch)

    (images, targets), records = batch
    for target in targets:
        assert set(target) == {"labels", "boxes"}


def _test_infer_batch(batch):
    assert len(batch) == 1

    batch = batch[0]
    assert len(batch) == 2

    for x in batch:
        assert x.shape == (3, 4, 4)
        assert x.dtype == torch.float


def test_faster_rcnn_build_train_batch(records):
    batch = faster_rcnn.build_train_batch(records)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_build_train_batch_empty(img):
    records = [{"img": img, "labels": [], "bboxes": []}] * 2
    (_, targets), _ = faster_rcnn.build_train_batch(records)

    for x in targets:
        assert x["labels"].equal(torch.zeros(0, dtype=torch.int64))
        assert x["boxes"].equal(torch.zeros((0, 4), dtype=torch.float32))


def test_faster_rcnn_build_valid_batch(records):
    batch = faster_rcnn.build_valid_batch(records)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_build_infer_batch(dataset):
    batch, samples = faster_rcnn.build_infer_batch(dataset)
    _test_infer_batch(batch)


def test_faster_rcnn_train_dataloader(records):
    dl = faster_rcnn.train_dl(records, batch_size=2)
    batch = first(dl)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_valid_dataloader(records):
    dl = faster_rcnn.valid_dl(records, batch_size=2)
    batch = first(dl)
    _test_faster_rcnn_batch(batch=batch)


def test_faster_rcnn_infer_dataloader(dataset):
    dl = faster_rcnn.infer_dl(dataset=dataset, batch_size=2)
    batch, samples = first(dl)
    _test_infer_batch(batch=batch)


### Mask RCNN ###
@pytest.fixture()
def masks():
    return MaskArray(np.ones((2, 4, 4), dtype=np.uint8))


# TODO: Duplication with fasterrcnn records
@pytest.fixture()
def mask_records(img, labels, bboxes, masks):
    Record = create_mixed_record(
        (ImageRecordMixin, LabelsRecordMixin, BBoxesRecordMixin, MasksRecordMixin)
    )
    record = Record()
    record.set_imageid(1)
    record.set_img(img)
    record.add_labels(labels)
    record.add_bboxes(bboxes)
    record.add_masks([masks])
    sample = record.load()
    return [sample] * 2


def _test_mask_rcnn_batch(batch):
    _test_common_rcnn_batch(batch=batch)
    (images, targets), records = batch

    for target in targets:
        assert set(target.keys()) == {"labels", "boxes", "masks"}
        assert target["masks"].shape == (2, 4, 4)
        assert target["masks"].dtype == torch.uint8


def test_mask_rcnn_build_train_batch(mask_records):
    batch = mask_rcnn.build_train_batch(mask_records)
    _test_mask_rcnn_batch(batch)


def test_mask_rcnn_build_train_batch_empty(img):
    records = [{"img": img, "labels": [], "bboxes": [], "masks": []}] * 2
    (_, targets), _ = mask_rcnn.build_train_batch(records)

    height, width = img.shape[:-1]
    for x in targets:
        assert x["labels"].equal(torch.zeros(0, dtype=torch.int64))
        assert x["boxes"].equal(torch.zeros((0, 4), dtype=torch.float32))
        assert x["masks"].equal(torch.zeros((0, height, width), dtype=torch.uint8))


def test_mask_rcnn_build_valid_batch(mask_records):
    batch = mask_rcnn.build_valid_batch(mask_records)
    _test_mask_rcnn_batch(batch)


def test_mask_rcnn_build_infer_batch(dataset):
    test_faster_rcnn_infer_dataloader(dataset)


def test_mask_rcnn_train_dataloader(mask_records):
    dl = mask_rcnn.train_dl(mask_records, batch_size=2)
    batch = first(dl)
    _test_mask_rcnn_batch(batch=batch)


def test_mask_rcnn_valid_dataloader(mask_records):
    dl = mask_rcnn.valid_dl(mask_records, batch_size=2)
    batch = first(dl)
    _test_mask_rcnn_batch(batch=batch)


def test_mask_rcnn_infer_dataloader(dataset):
    test_faster_rcnn_infer_dataloader(dataset)
