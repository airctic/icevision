import pytest
import mantisshrimp
from mantisshrimp.models.mantis_rcnn import *
from torchvision.transforms.functional import to_tensor as im2tensor
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import requests
import numpy as np
import cv2
from PIL import Image

# Passing a custom CNN as a Backbone should be supported


def get_image():
    # Get a big image because of these big CNNs
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = np.array(Image.open(requests.get(url, stream=True).raw))
    # Get a big size image for these big resnets
    img = cv2.resize(img, (2048, 2048))
    tensor_img = im2tensor(img)
    tensor_img = torch.unsqueeze(tensor_img, 0)
    return tensor_img


# Just pass a resnet18 as if user wrote it


def test_custom_backbone():
    backbone = torchvision.models.resnet18(pretrained=False)
    model = MantisFasterRCNN(n_class=10, backbone=backbone, out_channels=512)
    model.eval()
    print("Testing custom backbone")
    image = get_image()
    pred = model(image)
    assert isinstance(pred, list)
