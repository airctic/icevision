from icevision.parsers.coco_parser import COCOKeypointsMetadata
import pytest
from icevision.all import *


def test_keypoints_simple(keypoints_img_128372):
    kps = KeyPoints.from_xyv(keypoints_img_128372, COCOKeypointsMetadata)

    assert len(kps.keypoints) == 51
    assert len(kps.metadata.connections) == 19
    assert all([isinstance(o, KeypointConnection) for o in kps.metadata.connections])
    assert len(kps.metadata.labels) == 17
    assert kps.metadata.labels[0] == "nose"
    assert (kps.visible == np.array(keypoints_img_128372[2::3])).all()
    assert (kps.x == np.array(keypoints_img_128372[0::3])).all()
    assert kps.xy == [(x, y) for x, y in zip(kps.x, kps.y)]
    assert kps.xy[0] == (0, 0)
