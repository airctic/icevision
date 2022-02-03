__all__ = [
    "r50_fpn_1x_coco",
    "r50_fpn_gn_1x_coco",
    "r101_fpn_1x_coco",
    "r101_fpn_gn_1x_coco",
    "r101_fpn_gn_2x_ms_640_800_coco",
    "r101_fpn_gn_2x_ms_480_960_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetSABLBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="sabl", **kwargs)


base_config_path = mmdet_configs_path / "sabl"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/sabl"

r50_fpn_1x_coco = MMDetSABLBackboneConfig(
    config_path=base_config_path / "sabl_retinanet_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/sabl_retinanet_r50_fpn_1x_coco/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth",
)

r50_fpn_gn_1x_coco = MMDetSABLBackboneConfig(
    config_path=base_config_path / "sabl_retinanet_r50_fpn_gn_1x_coco.py",
    weights_url=f"{base_weights_url}/sabl_retinanet_r50_fpn_gn_1x_coco/sabl_retinanet_r50_fpn_gn_1x_coco-e16dfcf1.pth",
)

r101_fpn_1x_coco = MMDetSABLBackboneConfig(
    config_path=base_config_path / "sabl_retinanet_r101_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/sabl_retinanet_r101_fpn_1x_coco/sabl_retinanet_r101_fpn_1x_coco-42026904.pth",
)

r101_fpn_gn_1x_coco = MMDetSABLBackboneConfig(
    config_path=base_config_path / "sabl_retinanet_r101_fpn_gn_1x_coco.py",
    weights_url=f"{base_weights_url}/sabl_retinanet_r101_fpn_gn_1x_coco/sabl_retinanet_r101_fpn_gn_1x_coco-40a893e8.pth",
)

r101_fpn_gn_2x_ms_640_800_coco = MMDetSABLBackboneConfig(
    config_path=base_config_path / "sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco.py",
    weights_url=f"{base_weights_url}/sabl_retinanet_r50_fpn_gn_1x_coco/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco-1e63382c.pth",
)

r101_fpn_gn_2x_ms_480_960_coco = MMDetSABLBackboneConfig(
    config_path=base_config_path / "sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco.py",
    weights_url=f"{base_weights_url}/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco-5342f857.pth",
)
