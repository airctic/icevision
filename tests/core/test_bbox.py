import pytest
from mantisshrimp import *


def test_bbox_relative_xcycwh():
    w, h = 640, 480
    xcycwh = [0.7, 0.2, 0.1, 0.2]
    bbox = BBox.from_relative_xcycwh(*xcycwh, img_w=w, img_h=h)
    assert bbox.xyxy == [416, 48, 480, 144]
    assert bbox.relative_xcycwh == pytest.approx(xcycwh)
