__all__ = [
    # "mobilenetv2_100",
    # "mobilenetv2_110d",
    # "mobilenetv2_120d",
    # "mobilenetv2_140",
    "mobilenetv3_large_075",
    "mobilenetv3_large_100",
    "mobilenetv3_rw",
    "mobilenetv3_small_075",
    "mobilenetv3_small_100",
    "tf_mobilenetv3_large_075",
    "tf_mobilenetv3_large_100",
    "tf_mobilenetv3_large_minimal_100",
    "tf_mobilenetv3_small_075",
    "tf_mobilenetv3_small_100",
    "tf_mobilenetv3_small_minimal_100",
]

from icevision.models.mmdet.models.retinanet.backbones.timm.common import *
from icevision.models.mmdet.backbones.timm.mobilenet import *

from icevision.imports import *
from icevision.models.mmdet.utils import *

# mobilenetv2_100 = MMDetTimmRetinanetBackboneConfig(
#     backbone_dict={
#         "type": "MobileNetV2_100",
#         "pretrained": True,
#         "out_indices": (0, 1, 2, 3, 4),
#     },
# )

# mobilenetv2_110d = MMDetTimmRetinanetBackboneConfig(
#     backbone_dict={
#         "type": "MobileNetV2_110D",
#         "pretrained": True,
#         "out_indices": (0, 1, 2, 3, 4),
#     },
# )

# mobilenetv2_140 = MMDetTimmRetinanetBackboneConfig(
#     backbone_dict={
#         "type": "MobileNetV2_140",
#         "pretrained": True,
#         "out_indices": (0, 1, 2, 3, 4),
#     },
# )

mobilenetv3_large_075 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "MobileNetV3_Large_075",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

mobilenetv3_large_100 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "MobileNetV3_Large_100",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
        "init_cfg": {
            "type": "Pretrained",
            "checkpoint": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth",
        },
    },
)

mobilenetv3_rw = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "MobileNetV3_RW",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

mobilenetv3_small_075 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "MobileNetV3_Small_075",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

mobilenetv3_small_100 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "MobileNetV3_Small_100",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

tf_mobilenetv3_large_075 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "TF_MobileNetV3_Large_075",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

tf_mobilenetv3_large_100 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "TF_MobileNetV3_Large_100",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

tf_mobilenetv3_large_minimal_100 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "TF_MobileNetV3_Large_Minimal_100",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

tf_mobilenetv3_small_075 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "TF_MobileNetV3_Small_075",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

tf_mobilenetv3_small_100 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "TF_MobileNetV3_Small_100",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)

tf_mobilenetv3_small_minimal_100 = MMDetTimmRetinanetBackboneConfig(
    backbone_dict={
        "type": "TF_MobileNetV3_Small_Minimal_100",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)
