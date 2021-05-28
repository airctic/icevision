__all__ = [
    "resnet50",

]

from icevision.models.mmdet.models.retinanet.backbones.timm.common import *
from icevision.models.mmdet.backbones.timm.resnet import *

from icevision.imports import *
from icevision.models.mmdet.utils import *


resnet50 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNet50_TIMM",
        "pretrained": True,
        "out_indices": (2, 3, 4),
    },
    weights_url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
)
