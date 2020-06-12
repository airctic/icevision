import pytest
import mantisshrimp
from mantisshrimp.models.mantis_rcnn import *
from torchvision.transforms.functional import to_tensor as im2tensor
import torch
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_custom_backbone():
    backbone = Net()
    model = MantisFasterRCNN(n_class=10, backbone=backbone)
    model.eval()
    print("Testing custom backbone")
    image = get_image()
    pred = model(image)
    assert isinstance(pred, list)
