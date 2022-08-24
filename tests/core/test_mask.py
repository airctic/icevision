import pytest
from icevision.all import *


@pytest.fixture()
def coco_counts():
    # decoded: 1 1 1 0 0 0 1 1 0 0 0 0 0 0 1 0 0
    return [0, 3, 3, 2, 6, 1, 2]


@pytest.fixture
def kaggle_counts():
    # TODO: I originally thought the second to last number should've been 17
    # but .to_coco seems to work correctly only with 18, why?
    return [1, 3, 7, 2, 15, 1, 18, 0]


def test_rle_from_kaggle(kaggle_counts, coco_counts):
    rle = RLE.from_kaggle(kaggle_counts)
    assert rle.to_coco() == coco_counts


def test_rle_to_coco(coco_counts):
    rle = RLE.from_coco(coco_counts)
    assert rle.to_coco() == coco_counts


def test_rle_to_mask(coco_counts):
    rle = RLE.from_coco(coco_counts)
    mask = rle.to_mask(17, 1)
    expected = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert mask.data.reshape(-1).tolist() == expected


def test_rle_to_erle(coco_counts):
    rle = RLE.from_coco(coco_counts)

    coco_erles = rle.to_erles(17, 1)
    coco_rle = coco_erles.to_mask(17, 1).to_coco_rle(17, 1)

    assert coco_rle[0]["counts"] == rle.to_coco()


def test_mask_array_to_coco_rle():
    mask = MaskArray(np.array([[[0, 0, 1, 1, 1, 0, 1]]]))
    assert mask.to_coco_rle(h=1, w=7) == [{"counts": [2, 3, 1, 1], "size": (1, 7)}]

    mask = MaskArray(np.array([[[1, 1, 1, 1, 1, 1, 0]]]))
    assert mask.to_coco_rle(h=1, w=7) == [{"counts": [0, 6, 1], "size": (1, 7)}]


def test_voc_mask_file(samples_source):
    mask_filepath = samples_source / "voc/SegmentationObject/2007_000063.png"
    mask = VocMaskFile(mask_filepath)

    mask_arr = mask.to_mask(0, 0)
    assert mask_arr.data.shape == (2, 375, 500)

    erles = mask.to_erles(h=375, w=500)
    assert len(erles) == 2


# TODO: Test that all types of masks return the same encoded RLE
def test_mask_encode_decode(samples_source):
    mask_filepath = samples_source / "voc/SegmentationObject/2007_000063.png"
    mask = VocMaskFile(mask_filepath)

    erle = mask.to_erles(0, 0)
    mask_decoded = erle.to_mask(0, 0)

    mask_expected = mask.to_mask(0, 0)
    np.testing.assert_equal(mask_decoded.data, mask_expected.data)


def test_mask_unconnected_polygon(samples_source):
    annotations_dict = json.loads((samples_source / "annotations.json").read_bytes())
    annotation = [o for o in annotations_dict["annotations"] if o["id"] == 257712][0]

    poly = Polygon(annotation["segmentation"])

    h, w = 427, 640
    mask = poly.to_erles(h, w).to_mask(h, w)

    assert mask.shape == (1, h, w)
