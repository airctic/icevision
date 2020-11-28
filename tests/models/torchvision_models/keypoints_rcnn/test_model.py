from icevision.all import *


def test_keypoints_rcnn_model():
    model = keypoint_rcnn.model(num_keypoints=1)
    param_groups = model.param_groups()

    assert len(param_groups) == 8
    assert model.roi_heads.keypoint_predictor.kps_score_lowres.out_channels == 1

    backbone = backbones.resnet_fpn.resnet18(pretrained=True)
    model = keypoint_rcnn.model(backbone=backbone, num_keypoints=10)

    assert len(param_groups) == 8
    assert model.roi_heads.keypoint_predictor.kps_score_lowres.out_channels == 10
