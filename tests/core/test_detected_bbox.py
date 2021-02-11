import pytest
from icevision.all import *


def test_bbox_simple():
    bbox = DetectedBBox.from_xyxy(1, 2, 3, 4, 1.0, 10)

    assert bbox.xyxy == (1, 2, 3, 4)
    assert bbox.yxyx == (2, 1, 4, 3)
    assert bbox.xywh == (1, 2, 2, 2)
    assert bbox.score == 1.0
    assert bbox.label == 10


def test_bbox_relative_xcycwh():
    w, h = 640, 480
    xcycwh = (0.7, 0.2, 0.1, 0.2)
    bbox = DetectedBBox.from_relative_xcycwh(
        *xcycwh, img_width=w, img_height=h, score=1.0, label=10
    )
    assert bbox.xyxy == (416, 48, 480, 144)
    assert bbox.relative_xcycwh(img_width=w, img_height=h) == pytest.approx(xcycwh)


@pytest.mark.parametrize("xyxy", [(1, 2, 2, 2), (-2, 1, -1, 2), (3, 1, 4, 2)])
def test_bbox_autofix_invalid_data_error(xyxy):
    bbox = DetectedBBox.from_xyxy(*xyxy, 1.0, 1)
    with pytest.raises(InvalidDataError) as e:
        bbox.autofix(2, 2)


def test_bbox_autofix():
    bbox = DetectedBBox.from_xyxy(-1, 1, 4, 4, 1.0, -1)
    bbox.autofix(img_w=3, img_h=2)
    assert bbox.xyxy == (0, 1, 3, 2)
