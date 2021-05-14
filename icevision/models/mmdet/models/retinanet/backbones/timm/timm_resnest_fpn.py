__all__ = ["resnest14d",
    "resnest26d",
    "resnest50d",
    "resnest50d_1s4x24d",
    "resnest50d_4s2x40d",
    "resnest101e",
    "resnest200e",
    "resnest269e"
 ]

from icevision.models.mmdet.models.retinanet.backbones.timm.common import *


resnest14d = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNest14D",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

resnest26d = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNest26D",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

resnest50d = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNest50D",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

"resnest50d_1s4x24d = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNest50D_1S4x24D",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

resnest50d_4s2x40d = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNest50D_4S2x40D",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

resnest101e = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNest101E",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

resnest269e = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "ResNest269E",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

