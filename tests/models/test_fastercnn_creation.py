import pytest
import mantisshrimp
from mantisshrimp.models.mantis_rcnn import *
import torch


def test_fastercnn():
    # We need to instantiate with all possible combinations
    # Taken directly from mantis_faster_rcnn
    supported_resnet_fpn_models = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ]

    # The remaining we need to add the layer extraction in torchvision backbones.
    # I hope it is similar. But for now.

    supported_non_fpn_models = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        # "resnext50_32x4d",
        "resnext101_32x8d",
        # "wide_resnet50_2",
        # "wide_resnet101_2",
        "mobilenet",
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
    ]

    # First check for none backbone
    backbone_ = None
    is_pretrained = False
    is_fpn = False
    num_classes = 3
    model = MantisFasterRCNN(
        n_class=num_classes, backbone=backbone_, pretrained=is_pretrained, fpn=is_fpn
    )
    assert isinstance(model, mantisshrimp.models.mantis_faster_rcnn.MantisFasterRCNN)

    # Else try all the resnet models with fpns
    for backbone_ in supported_resnet_fpn_models:
        is_fpn = True
        model = MantisFasterRCNN(
            n_class=num_classes,
            backbone=backbone_,
            pretrained=is_pretrained,
            fpn=is_fpn,
        )
        assert isinstance(
            model, mantisshrimp.models.mantis_faster_rcnn.MantisFasterRCNN
        )

    # Now test instantiating for non fpn models
    is_fpn = False
    for backbone_ in supported_non_fpn_models:
        is_fpn = False
        model = MantisFasterRCNN(
            n_class=num_classes,
            backbone=backbone_,
            pretrained=is_pretrained,
            fpn=is_fpn,
        )
        assert isinstance(
            model, mantisshrimp.models.mantis_faster_rcnn.MantisFasterRCNN
        )

    # Check for simple CNN that can be passed to make backbone
    # Will think of an example and add soon

    # Stuff to be added, running these models on a fake image data.
