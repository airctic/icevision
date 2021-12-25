__all__ = [
    "yolox_tiny_8x8",
    "yolox_s_8x8",
    "yolox_l_8x8",
    "yolox_x_8x8",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetYOLOXBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="yolox", **kwargs)


base_config_path = mmdet_configs_path / "yolox"
base_weights_url = "https://download.openmmlab.com/mmdetection/v2.0/yolox/"

yolox_tiny_8x8 = MMDetYOLOXBackboneConfig(
    config_path=base_config_path / "yolox_tiny_8x8_300e_coco.py",
    weights_url=f"{base_weights_url}yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth",
)

yolox_s_8x8 = MMDetYOLOXBackboneConfig(
    config_path=base_config_path / "yolox_s_8x8_300e_coco.py",
    weights_url=f"{base_weights_url}yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
)

yolox_l_8x8 = MMDetYOLOXBackboneConfig(
    config_path=base_config_path / "yolox_l_8x8_300e_coco.py",
    weights_url=f"{base_weights_url}yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
)

yolox_x_8x8 = MMDetYOLOXBackboneConfig(
    config_path=base_config_path / "yolox_x_8x8_300e_coco.py",
    weights_url=f"{base_weights_url}yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
)
