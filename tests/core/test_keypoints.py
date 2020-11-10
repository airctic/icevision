import pytest
from icevision.all import *


def test_keypoints_simple(keypoints_img_128372):
    kps = KeyPoints.from_xyv(keypoints_img_128372)

    assert (kps.human_conns[0] == np.array([15, 13])).all()
    assert len(kps.keypoints) == 51
    assert len(kps.human_kps) == 17
    assert kps.human_kps[0] == "nose"
    assert (kps.visible == np.array(keypoints_img_128372[2::3])).all()
    assert (kps.x == np.array(keypoints_img_128372[0::3])).all()
    assert kps.xy == [(x, y) for x, y in zip(kps.x, kps.y)]
    assert kps.xy[0] == (0, 0)
