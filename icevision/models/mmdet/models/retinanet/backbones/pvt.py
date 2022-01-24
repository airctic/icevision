__all__ = [
    "retinanet_pvtv2_b0_fpn_1x_coco",
    "retinanet_pvtv2_b1_fpn_1x_coco",
    "retinanet_pvtv2_b2_fpn_1x_coco",
    "retinanet_pvtv2_b3_fpn_1x_coco",
    "retinanet_pvtv2_b4_fpn_1x_coco",
    "retinanet_pvtv2_b5_fpn_1x_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetPVTBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="pvt", **kwargs)


base_config_path = mmdet_configs_path / "pvt"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/pvt"

retinanet_pvtv2_b0_fpn_1x_coco = MMDetPVTBackboneConfig(
    config_path=base_config_path / "retinanet_pvtv2-b0_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/retinanet_pvtv2-b5_fpn_1x_coco/retinanet_pvtv2-b0_fpn_1x_coco/retinanet_pvtv2-b0_fpn_1x_coco_20210831_103157-13e9aabe.pth",
)

retinanet_pvtv2_b1_fpn_1x_coco = MMDetPVTBackboneConfig(
    config_path=base_config_path / "retinanet_pvtv2-b1_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/retinanet_pvtv2-b1_fpn_1x_coco/retinanet_pvtv2-b1_fpn_1x_coco_20210831_103318-7e169a7d.pth",
)

retinanet_pvtv2_b2_fpn_1x_coco = MMDetPVTBackboneConfig(
    config_path=base_config_path / "retinanet_pvtv2-b2_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/retinanet_pvtv2-b2_fpn_1x_coco/retinanet_pvtv2-b2_fpn_1x_coco_20210901_174843-529f0b9a.pth",
)

retinanet_pvtv2_b3_fpn_1x_coco = MMDetPVTBackboneConfig(
    config_path=base_config_path / "retinanet_pvtv2-b3_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/retinanet_pvtv2-b3_fpn_1x_coco/retinanet_pvtv2-b3_fpn_1x_coco_20210903_151512-8357deff.pth",
)

retinanet_pvtv2_b4_fpn_1x_coco = MMDetPVTBackboneConfig(
    config_path=base_config_path / "retinanet_pvtv2-b4_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/retinanet_pvtv2-b4_fpn_1x_coco/retinanet_pvtv2-b4_fpn_1x_coco_20210901_170151-83795c86.pth",
)

retinanet_pvtv2_b5_fpn_1x_coco = MMDetPVTBackboneConfig(
    config_path=base_config_path / "retinanet_pvtv2-b5_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/retinanet_pvtv2-b5_fpn_1x_coco/retinanet_pvtv2-b5_fpn_1x_coco_20210902_201800-3420eb57.pth",
)
