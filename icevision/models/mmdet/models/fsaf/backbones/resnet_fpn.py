__all__ = ["r50_fpn_1x_coco", "r101_fpn_1x_coco", "x101_64x4d_fpn_1x_coco"]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetFSAFBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="fsaf", **kwargs)


base_config_path = mmdet_configs_path / "fsaf"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/fsaf"

r50_fpn_1x_coco = MMDetFSAFBackboneConfig(
    config_path=base_config_path / "fsaf_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth",
)

r101_fpn_1x_coco = MMDetFSAFBackboneConfig(
    config_path=base_config_path / "fsaf_r101_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/fsaf_r101_fpn_1x_coco/fsaf_r101_fpn_1x_coco-9e71098f.pth",
)

x101_64x4d_fpn_1x_coco = MMDetFSAFBackboneConfig(
    config_path=base_config_path / "fsaf_x101_64x4d_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth",
)
