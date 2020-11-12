import pytest
from icevision.all import *


# TODO: Duplicated fixture between here and torchvision_models:test_dataloaders
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


def _test_batch(images, targets):
    assert images.shape == (2, 3, 4, 4)
    assert torch.all(images == 1.0)

    assert targets["cls"][0].dtype == torch.float
    assert len(targets["cls"]) == 2
    assert torch.all(targets["cls"][0] == tensor([1, 2], dtype=torch.float))

    assert targets["bbox"][0].dtype == torch.float
    assert len(targets["bbox"]) == 2
    expected_bboxes = tensor([[2, 1, 4, 3], [20, 10, 40, 30]], dtype=torch.float)
    assert torch.all(targets["bbox"][0] == expected_bboxes)


def _test_batch_train(images, targets):
    _test_batch(images=images, targets=targets)
    assert set(targets.keys()) == {"cls", "bbox"}


def _test_batch_valid(images, targets):
    _test_batch(images=images, targets=targets)

    assert set(targets.keys()) == {"cls", "bbox", "img_size", "img_scale"}

    assert targets["img_scale"].dtype == torch.float
    assert torch.all(targets["img_scale"] == tensor([1, 1]))

    assert targets["img_size"].dtype == torch.float
    assert torch.all(targets["img_size"] == tensor([[4, 4], [4, 4]]))


def test_efficient_det_build_train_batch(records):
    (images, targets), records = efficientdet.build_train_batch(records)
    _test_batch_train(images=images, targets=targets)


def test_efficient_det_build_valid_batch(records):
    (images, targets), records = efficientdet.build_valid_batch(records)
    _test_batch_valid(images=images, targets=targets)


def test_efficient_det_train_dataloader(records):
    dl = efficientdet.train_dl(records, batch_size=2)
    (xb, yb), records = first(dl)

    _test_batch_train(images=xb, targets=yb)


def test_efficient_det_valid_dataloader(records):
    dl = efficientdet.valid_dl(records, batch_size=2)
    (xb, yb), records = first(dl)

    _test_batch_valid(images=xb, targets=yb)


@pytest.mark.parametrize(
    "batch_tfms", [None, tfms.batch.ImgPadStack(np.array(0, dtype=np.uint8))]
)
def test_efficient_det_build_infer_batch(img, batch_tfms):
    records = [{"img": img, "height": 4, "width": 4}] * 2
    batch, records = efficientdet.build_infer_batch(records, batch_tfms=batch_tfms)

    tensor_img = torch.stack([im2tensor(img), im2tensor(img)])
    img_sizes = tensor([(4, 4), (4, 4)], dtype=torch.float)
    img_scales = tensor([1, 1], dtype=torch.float)
    img_info = {"img_size": img_sizes, "img_scale": img_scales}

    batch_imgs, batch_info = batch
    assert torch.equal(batch_imgs, tensor_img)
    assert torch.equal(batch_info["img_size"], img_info["img_size"])
    assert torch.equal(batch_info["img_scale"], img_info["img_scale"])
