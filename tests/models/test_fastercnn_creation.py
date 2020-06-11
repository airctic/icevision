import pytest
import mantisshrimp
from mantisshrimp.models.mantis_rcnn import *
from torchvision.transforms.functional import to_tensor as im2tensor
import torch
import requests
import numpy as np
import cv2
from PIL import Image


def get_image():
    # Get a big image because of these big CNNs
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = np.array(Image.open(requests.get(url, stream=True).raw))
    # Get a big size image for these big resnets
    img = cv2.resize(img, (2048, 2048))
    tensor_img = im2tensor(img)
    tensor_img = torch.unsqueeze(tensor_img, 0)
    return tensor_img


image = get_image()


def test_fastercnn():
    # We need to instantiate with all possible combinations
    # Taken directly from mantis_faster_rcnn
    supported_resnet_fpn_models = [
        "resnet18",
        "resnet34",
        "resnet50",
        # "resnet101",
        # "resnet152",
        # "resnext50_32x4d",
        # "resnext101_32x8d",
        # "wide_resnet50_2",
        # "wide_resnet101_2",
    ]

    # The remaining we need to add the layer extraction in torchvision backbones.
    # I hope it is similar. But for now.

    supported_non_fpn_models = [
        "resnet18",
        "resnet34",
        "resnet50",
        # "resnet101",
        # "resnet152",
        # "resnext50_32x4d",
        # "resnext101_32x8d",
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
    model.eval()
    pred = model(image)
    assert isinstance(pred, list)

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
        model.eval()
        print("Testing backbone = {}".format(backbone_))
        pred = model(image)
        assert isinstance(pred, list)

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
        model.eval()
        print("Testing backbone = {}".format(backbone_))
        pred = model(image)
        assert isinstance(pred, list)

    # Check for simple CNN that can be passed to make backbone
    # Will think of an example and add soon

    # Stuff to be added, running these models on a fake image data.
