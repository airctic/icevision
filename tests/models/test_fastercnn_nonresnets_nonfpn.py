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


@pytest.mark.slow
@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize(
    "backbone",
    [
        "mobilenet",
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext101_32x8d",
    ],
)
def test_faster_rcnn_backbones(image, backbone, pretrained):
    backbone = MantisFasterRCNN.get_backbone_by_name(
        name=backbone, fpn=False, pretrained=pretrained
    )
    model = MantisFasterRCNN(n_class=3, backbone=backbone)
    model.eval()
    pred = model(image)
    assert isinstance(pred, list)
    assert set(["boxes", "labels", "scores"]) == set(pred[0].keys())
