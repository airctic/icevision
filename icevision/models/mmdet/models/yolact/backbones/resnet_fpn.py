__all__ = ["r50_1x8_coco", "r50_8x8_coco", "r101_1x8_coco"]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class YOLOACTMMDetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="yolact", **kwargs)


base_config_path = mmdet_configs_path / "yolact"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/yolact"

r50_1x8_coco = YOLOACTMMDetBackboneConfig(
    config_path=base_config_path / "yolact_r50_1x8_coco.py",
    weights_url=f"{base_weights_url}/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth",
)

r50_8x8_coco = YOLOACTMMDetBackboneConfig(
    config_path=base_config_path / "yolact_r50_8x8_coco.py",
    weights_url=f"{base_weights_url}/yolact_r50_8x8_coco/yolact_r50_8x8_coco_20200908-ca34f5db.pth",
)

r101_1x8_coco = YOLOACTMMDetBackboneConfig(
    config_path=base_config_path / "yolact_r101_1x8_coco.py",
    weights_url=f"{base_weights_url}/yolact_r101_1x8_coco/yolact_r101_1x8_coco_20200908-4cbe9101.pth",
)
