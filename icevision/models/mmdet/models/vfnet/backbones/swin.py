__all__ = [
    "swin_t_p4_w7_fpn_1x_coco",
    "swin_s_p4_w7_fpn_1x_coco",
    "swin_b_p4_w7_fpn_1x_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *
from urllib.request import urlopen, urlcleanup
from icevision.models.mmdet.models.vfnet.backbones.backbone_config import (
    MMDetVFNETBackboneConfig,
)

base_config_path = mmdet_configs_path.parent / "custom_configs" / "vfnet"

swin_t_p4_w7_fpn_1x_coco = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_swin-t-p4-w7_fpn_1x_coco.py",
    weights_url="",  # There are no pretrained weights available for this model
)

swin_s_p4_w7_fpn_1x_coco = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_swin-s-p4-w7_fpn_1x_coco.py",
    weights_url="",  # There are no pretrained weights available for this model
)

swin_b_p4_w7_fpn_1x_coco = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_swin-b-p4-w7_fpn_1x_coco.py",
    weights_url="",  # There are no pretrained weights available for this model
)
