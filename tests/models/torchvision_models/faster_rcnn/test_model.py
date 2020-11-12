from icevision.all import *


def test_faster_rcnn_default_param_groups():
    model = faster_rcnn.model(num_classes=4)

    param_groups = model.param_groups()
    assert len(param_groups) == 8


def test_faster_rcnn_mobile_param_groups():
    backbone = backbones.mobilenet(pretrained=False)
    model = faster_rcnn.model(num_classes=6, backbone=backbone)

    param_groups = model.param_groups()
    assert len(param_groups) == 6
