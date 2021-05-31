__all__ = [
    "resnest50d",
]

from icevision.models.mmdet.models.retinanet.backbones.timm.common import *
from icevision.models.mmdet.backbones.timm.resnet import *

from icevision.imports import *
from icevision.models.mmdet.utils import *


resnest50d = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNest50D_TIMM",
        "pretrained": True,
        "out_indices": (2, 3, 4),
        "norm_eval": True,
        "frozen_stem": True,
        "frozen_stages": 1,
    },
    weights_url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth",
)
