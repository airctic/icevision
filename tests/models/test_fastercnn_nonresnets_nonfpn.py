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


def test_fastercnn_nonresnets_nonfpn():
    # We need to instantiate with all possible combinations
    # Taken directly from mantis_faster_rcnn
    # The remaining we need to add the layer extraction in torchvision backbones.
    # I hope it is similar. But for now.

    supported_non_fpn_models = [
        "mobilenet",  # Passing
        "vgg11",  # Passing
        "vgg13",  # Passing
        "vgg16",  # Passing
        "vgg19",  # Passing
        # "resnet18",  # Passing
        # "resnet34",  # Passing
        # "resnet50",  # Passing
        # "resnet101",
        # "resnet152",
        ## "resnext50_32x4d",
        # "resnext101_32x8d",
        ## "wide_resnet50_2",
        ## "wide_resnet101_2",
    ]

    # The default one

    backbone_ = None
    # is_pretrained = False
    num_classes = 3

    ## DEFAULT CASE CHECK

    for is_pretrained in [False]:
        model = MantisFasterRCNN(n_class=num_classes, backbone=None,)
        assert isinstance(
            model, mantisshrimp.models.mantis_faster_rcnn.MantisFasterRCNN
        )
        model.eval()
        image = get_image()
        pred = model(image)
        assert isinstance(pred, list)

    # First check for none backbone
    num_classes = 3
    # Now test instantiating for non fpn models
    is_fpn = False
    for backbone_ in supported_non_fpn_models:
        for is_pretrained in [False]:
            is_fpn = False
            backbone = MantisFasterRCNN.get_backbone_by_name(
                name=backbone_, fpn=is_fpn, pretrained=is_pretrained
            )
            model = MantisFasterRCNN(n_class=num_classes, backbone=backbone,)
            assert isinstance(
                model, mantisshrimp.models.mantis_faster_rcnn.MantisFasterRCNN
            )
            model.eval()
            print("Testing backbone = {} {}".format(backbone_, is_pretrained))
            image = get_image()
            pred = model(image)
            assert isinstance(pred, list)

    # Check for simple CNN that can be passed to make backbone
    # Will think of an example and add soon

    # Stuff to be added, running these models on a fake image data.
