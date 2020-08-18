import pytest
from mantisshrimp.all import *


@pytest.fixture
def records():
    train_records, valid_records = test_utils.sample_records()
    return train_records


def test_simple_transform(records):
    tfm = tfms.A.Adapter([tfms.A.HorizontalFlip(p=1.0)])
    ds = Dataset(records)
    tfm_ds = Dataset(records, tfm=tfm)

    sample, tfmed = ds[0], tfm_ds[0]
    assert (tfmed["img"] == sample["img"][:, ::-1, :]).all()


def test_crop_transform(records):
    tfm = tfms.A.Adapter([tfms.A.CenterCrop(100, 100, p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)
    tfmed = tfm_ds[0]
    assert len(tfmed["labels"]) == 1
    assert len(tfmed["bboxes"]) == 1
    assert len(tfmed["masks"]) == 1
    assert len(tfmed["iscrowds"]) == 1


def test_crop_transform_empty(records):
    tfm = tfms.A.Adapter([tfms.A.Crop(0, 0, 100, 100, p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)
    tfmed = tfm_ds[0]
    assert len(tfmed["labels"]) == 0
    assert len(tfmed["bboxes"]) == 0
    assert len(tfmed["masks"]) == 0
    assert len(tfmed["iscrowds"]) == 0

