import pytest
from icevision.all import *


@pytest.fixture()
def simple_counts():
    # decoded: 1 1 1 0 0 0 1 1 0 0 0 0 0 0 1 0 0
    return [0, 3, 3, 2, 6, 1, 2]


def test_rle_from_coco(simple_counts):
    rle = RLE.from_coco(simple_counts)
    # TODO: I originally thought the second to last number should've been 17
    # but .to_coco seems to work correctly only with 18, why?
    assert rle.counts == [1, 3, 7, 2, 15, 1, 18, 0]


def test_rle_to_coco(simple_counts):
    rle = RLE.from_coco(simple_counts)
    assert rle.to_coco() == simple_counts


def test_rle_to_mask(simple_counts):
    rle = RLE.from_coco(simple_counts)
    mask = rle.to_mask(17, 1)
    expected = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert mask.data.reshape(-1).tolist() == expected


def test_mask_array_to_coco_rle():
    mask = MaskArray(np.array([[[0, 0, 1, 1, 1, 0, 1]]]))
    assert mask.to_coco_rle(h=1, w=7) == [{"counts": [2, 3, 1, 1], "size": (1, 7)}]

    mask = MaskArray(np.array([[[1, 1, 1, 1, 1, 1, 0]]]))
    assert mask.to_coco_rle(h=1, w=7) == [{"counts": [0, 6, 1], "size": (1, 7)}]


def test_voc_mask_file(samples_source):
    mask_filepath = samples_source / "voc/SegmentationObject/2007_000063.png"
    mask = VocMaskFile(mask_filepath)

    mask_arr = mask.to_mask(0, 0)
    assert mask_arr.data.shape[0] == 2
