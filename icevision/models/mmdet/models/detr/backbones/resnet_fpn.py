__all__ = [
    "r50_8x2_150e_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetDetrBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="detr", **kwargs)


base_config_path = mmdet_configs_path / "detr"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/detr"

r50_8x2_150e_coco = MMDetDetrBackboneConfig(
    config_path=base_config_path / "detr_r50_8x2_150e_coco.py",
    weights_url=f"{base_weights_url}/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth",
)
