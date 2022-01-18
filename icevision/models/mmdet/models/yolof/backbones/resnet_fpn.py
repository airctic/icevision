__all__ = ["yolof_r50_c5_8x8_1x_coco"]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetYOLOFBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="yolof", **kwargs)


base_config_path = mmdet_configs_path / "yolof"
base_weights_url = "https://download.openmmlab.com/mmdetection/v2.0/yolof/"

yolof_r50_c5_8x8_1x_coco = MMDetYOLOFBackboneConfig(
    config_path=base_config_path / "yolof_r50_c5_8x8_1x_coco.py",
    weights_url=f"{base_weights_url}yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth",
)
