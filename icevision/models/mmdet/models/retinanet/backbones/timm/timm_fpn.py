__all__ = [
    "mobilenetv3_large_100",
]

import icevision
from icevision.imports import *
from icevision.models.mmdet.utils import *

base_config_path = Path(icevision.__file__) / "models/mmdet/models/retinanet"


class MMDetTimmRetinanetBackboneConfig(MMDetTimmBackboneConfig):
    def __init__(self, backbone, **kwargs):
        super().__init__(
            model_name="retinanet",
            config_path=base_config_path / "model_dict_fpn_no_backbone.py",
            backbone=backbone,
            **kwargs
        )


mobilenetv3_large_100 = MMDetTimmRetinanetBackboneConfig(
    backbone={
        "type": "MobileNetv3Large100",
        "pretrained": True,
        "out_indices": (0, 1, 2, 3, 4),
    },
)
