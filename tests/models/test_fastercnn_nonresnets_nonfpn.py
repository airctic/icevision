import pytest
import mantisshrimp
from mantisshrimp.models.mantis_rcnn import *


@pytest.mark.slow
def test_fastercnn_default_backbone(image):
    model = MantisFasterRCNN(n_class=3)
    assert isinstance(model, mantisshrimp.models.mantis_faster_rcnn.MantisRCNN)
    model.eval()
    pred = model(image)
    assert isinstance(pred, list)
    assert set(["boxes", "labels", "scores"]) == set(pred[0].keys())


# @pytest.mark.skip(reason="exceeds 40 min limit on github CI")
@pytest.mark.slow
@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize(
    "backbone, fpn",
    [
        ("mobilenet", False),
        ("vgg11", False),
        ("vgg13", False),
        ("vgg16", False),
        ("vgg19", False),
        ("resnet18", False),
        ("resnet34", False),
        ("resnet50", False),
        ("resnet18", True),
        ("resnet34", True),
        ("resnet50", True),
        # these models are too big for github runners
        # "resnet101",
        # "resnet152",
        # "resnext101_32x8d",
    ],
)
def test_faster_rcnn_nonfpn_backbones(image, backbone, fpn, pretrained):
    backbone = MantisFasterRCNN.get_backbone_by_name(
        name=backbone, fpn=fpn, pretrained=pretrained
    )
    model = MantisFasterRCNN(n_class=3, backbone=backbone)
    model.eval()
    pred = model(image)
    assert isinstance(pred, list)
    assert set(["boxes", "labels", "scores"]) == set(pred[0].keys())
