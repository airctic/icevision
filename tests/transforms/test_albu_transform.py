import pytest
from mantisshrimp import *
import albumentations as A


@pytest.fixture
def records():
    train_records, valid_records = test_utils.sample_records()
    return train_records


def test_simple_transform(records):
    tfm = AlbuTransform([A.HorizontalFlip(p=1.0)])
    ds = Dataset(records)
    tfm_ds = Dataset(records, tfm=tfm)

    sample, tfmed = ds[0], tfm_ds[0]
    assert (tfmed["img"] == sample["img"][:, ::-1, :]).all()


def test_crop_transform(records):
    tfm = AlbuTransform([A.CenterCrop(100, 100, p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)
    tfmed = tfm_ds[0]
    print(tfmed["filepath"])
    assert len(tfmed["bboxes"]) == 1
    assert len(tfmed["iscrowds"]) == 1
