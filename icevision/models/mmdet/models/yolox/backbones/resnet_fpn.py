__all__ = [
    "yolox_tiny_8x8",
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
