import pytest
from mantisshrimp.core import *


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
