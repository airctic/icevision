__all__ = [
    "resnet18_140e_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetCenternetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="centernet", **kwargs)


base_config_path = mmdet_configs_path / "centernet"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/centernet"
)

resnet18_140e_coco = MMDetCenternetBackboneConfig(
    config_path=base_config_path / "centernet_resnet18_140e_coco.py",
    weights_url=f"{base_weights_url}/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth",
)
