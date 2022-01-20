__all__ = [
    "ssd300",
    "ssd512",
    "ssdlite_mobilenetv2",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetSSDBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="ssd", **kwargs)


base_config_path = mmdet_configs_path / "ssd"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/ssd"

ssd300 = MMDetSSDBackboneConfig(
    config_path=base_config_path / "ssd300_coco.py",
    weights_url=f"{base_weights_url}/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth",
)

ssd512 = MMDetSSDBackboneConfig(
    config_path=base_config_path / "ssd512_coco.py",
    weights_url=f"{base_weights_url}/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth",
)

ssdlite_mobilenetv2 = MMDetSSDBackboneConfig(
    config_path=base_config_path / "ssdlite_mobilenetv2_scratch_600e_coco.py",
    weights_url=f"{base_weights_url}/ssd512_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth",
)
