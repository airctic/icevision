import pytest
from icevision.all import *


@pytest.fixture
def records(coco_mask_records):
    return coco_mask_records


def test_inference_transform(records):
    img = open_img(records[0].filepath)
    tfm = tfms.A.Adapter([tfms.A.HorizontalFlip(p=1.0)])
    ds = Dataset.from_images([img], tfm)

    tfmed = ds[0]
    assert (tfmed.img == img[:, ::-1, :]).all()


def test_simple_transform(records):
    tfm = tfms.A.Adapter([tfms.A.HorizontalFlip(p=1.0)])
    ds = Dataset(records)
    tfm_ds = Dataset(records, tfm=tfm)

    sample, tfmed = ds[0], tfm_ds[0]
    assert (tfmed.img == sample.img[:, ::-1, :]).all()


def test_crop_transform(records):
    tfm = tfms.A.Adapter([tfms.A.CenterCrop(100, 100, p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)
    tfmed = tfm_ds[0]
    assert "keypoints" not in tfm_ds.tfm.tfms.processors.keys()
    assert len(tfmed.labels) == 1
    assert len(tfmed.bboxes) == 1
    assert len(tfmed.masks) == 1
    assert len(tfmed.iscrowds) == 1


def test_crop_transform_empty(records):
    tfm = tfms.A.Adapter([tfms.A.Crop(0, 0, 100, 100, p=1.0)])
    tfm_ds = Dataset(records, tfm=tfm)
    tfmed = tfm_ds[0]
    assert "keypoints" not in tfm_ds.tfm.tfms.processors.keys()
    assert len(tfmed.labels) == 0
    assert len(tfmed.bboxes) == 0
    assert len(tfmed.masks) == 0
    assert len(tfmed.iscrowds) == 0


def test_keypoints_transform(coco_keypoints_parser):
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    tfm = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=384, presize=512, crop_fn=None), tfms.A.Normalize()]
    )
    # assert "keypoints" in tfm.tfms.processors.keys()
    # assert "bboxes" in tfm.tfms.processors.keys()

    ds = Dataset(records)
    tfm_ds = Dataset(records, tfm=tfm)

    d, t = ds[0], tfm_ds[0]
    # assert "keypoints" in tfm.tfms.processors.keys()
    # assert "bboxes" in tfm.tfms.processors.keys()
    assert len(d.keypoints) == 3
    assert len(t.keypoints) == 3
    assert set([c for c in t.keypoints[0].visible]) == {0.0, 1.0, 2.0}
    assert set([c for c in d.keypoints[0].visible]) == {0, 1, 2}


def test_keypoints_transform_crop_error(coco_keypoints_parser):
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    tfm = tfms.A.Adapter([*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()])

    tfm_ds = Dataset(records, tfm=tfm)
    with pytest.raises(RuntimeError):
        tfm_ds[0]


def test_filter_keypoints():
    tfms_kps, w, h, v = (
        [(0, 0), (60, 119), (-30, 40), (100, 300), (30, 100)],
        80,
        120,
        [0, 1, 1, 1, 2],
    )
    img_size = ImgSize(width=w, height=h)
    tra_n = tfms.A.AlbumentationsKeypointsComponent._remove_albu_outside_keypoints(
        tfms_kps, v, img_size
    )

    assert len(tfms_kps) == len(tra_n)
    assert tra_n == [(0, 0, 0), (60, 119, 1), (0, 0, 0), (0, 0, 0), (30, 100, 2)]

    tfms_kps, w, h, v = (
        [(0, 0), (79, 119), (-30, 40), (100, 300), (70, 100)],
        120,
        120,
        [0, 1, 1, 1, 2],
    )
    tra_n = tfms.A.AlbumentationsKeypointsComponent._remove_albu_outside_keypoints(
        tfms_kps, v, img_size
    )

    assert len(tfms_kps) == len(tra_n)
    assert tra_n == [(0, 0, 0), (79, 119, 1), (0, 0, 0), (0, 0, 0), (70, 100, 2)]


def test_filter_boxes():
    inp = (52.17503641656451, 274.5014178489639, 123.51860681160832, 350.33471836373275)
    out = (52.17503641656451, 274.5014178489639, 123.51860681160832, 320)
    h, w = 256, 384

    res = tfms.A.AlbumentationsBBoxesComponent._clip_bboxes(inp, h, w)
    assert out == res
