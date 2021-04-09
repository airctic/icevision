from icevision import *
from icevision.models.torchvision import mask_rcnn


def test_mask_rcnn_default_param_groups():
    model = mask_rcnn.model(num_classes=4)

    param_groups = model.param_groups()
    assert len(param_groups) == 8


def test_mask_rcnn_mobile_param_groups():
    backbone = models.torchvision.mask_rcnn.backbones.mobilenet(pretrained=False)
    model = mask_rcnn.model(num_classes=6, backbone=backbone)

    param_groups = model.param_groups()
    assert len(param_groups) == 6
