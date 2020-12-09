import pytest
from icevision.all import *


def test_predict_keypoints_rcnn_train(ochuman_ds):
    _, valid_ds = ochuman_ds
    model = keypoint_rcnn.model(num_keypoints=19)

    infer_dl = keypoint_rcnn.infer_dl(valid_ds, batch_size=2)
    _, preds = keypoint_rcnn.predict_dl(model=model, infer_dl=infer_dl)
    p = preds[0]

    assert len(infer_dl) == 1
    assert len(preds) == 2
    assert len(list(p["keypoints_scores"])) == len(p["keypoints"])
    assert len(p["bboxes"]) == len(p["keypoints"])
