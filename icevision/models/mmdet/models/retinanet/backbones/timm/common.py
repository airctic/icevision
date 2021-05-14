__all__ = [
    "MMDetTimmRetinanetBackboneConfig",
]

import icevision
from icevision.imports import *
from icevision.models.mmdet.utils import *

base_config_path = Path(icevision.__file__).parent / "models/mmdet/models/retinanet"


class MMDetTimmRetinanetBackboneConfig(MMDetTimmBackboneConfig):
    def __init__(self, backbone_dict, **kwargs):
        super().__init__(
            model_name="retinanet",
            config_path=base_config_path / "retinanet_no_backbone_fpn.py",
            backbone_dict=backbone_dict,
            **kwargs
        )
