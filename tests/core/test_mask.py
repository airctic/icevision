from mantisshrimp.core import *


def test_rle_from_coco():
    rle = RLE.from_coco([0, 3, 3, 2, 6, 1])
    assert rle.counts == [1, 3, 7, 2, 15, 1]
