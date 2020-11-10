import pytest
from icevision.all import *


@pytest.fixture
def records(coco_mask_records):
    return coco_mask_records


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
    assert "keypoints" not in tfm.tfms.processors.keys()
    assert len(tfmed["labels"]) == 1
    assert len(tfmed["bboxes"]) == 1
    assert len(tfmed["masks"]) == 1
    assert len(tfmed["iscrowds"]) == 1


def test_crop_transform_empty(records):
    tfm = tfms.A.Adapter([tfms.A.Crop(0, 0, 100, 100, p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)
    tfmed = tfm_ds[0]
    assert "keypoints" not in tfm.tfms.processors.keys()
    assert len(tfmed["labels"]) == 0
    assert len(tfmed["bboxes"]) == 0
    assert len(tfmed["masks"]) == 0
    assert len(tfmed["iscrowds"]) == 0


def test_keypoints_transform(coco_keypoints_parser):
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    tfm = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=384, presize=512, crop_fn=None), tfms.A.Normalize()]
    )
    assert "keypoints" in tfm.tfms.processors.keys()
    assert "bboxes" in tfm.tfms.processors.keys()

    ds = Dataset(records)
    tfm_ds = Dataset(records, tfm=tfm)

    d, t = ds[0], tfm_ds[0]
    assert "keypoints" in tfm.tfms.processors.keys()
    assert "bboxes" not in tfm.tfms.processors.keys()
    assert len(d["keypoints"]) == 9
    assert len(t["keypoints"]) == 3
