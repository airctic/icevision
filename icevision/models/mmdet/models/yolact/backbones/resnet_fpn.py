__all__ = ["yolact_r50_1x8_coco"]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class YOLOACTMMDetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="yolact", **kwargs)


base_config_path = mmdet_configs_path / "yolact"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/yolact"

yolact_r50_1x8_coco = YOLOACTMMDetBackboneConfig(
    config_path=base_config_path / "yolact_r50_1x8_coco.py",
    weights_url=f"{base_weights_url}/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth",
)
