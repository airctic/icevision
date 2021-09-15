__all__ = [
    "resnet50_fpn_1x_coco",
    "resnet50_fpn_20e_coco",
    "resnet101_fpn_20e_coco ",
    "resnext101_64x4d_fpn_20e_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetscnetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="scnet", **kwargs)


base_config_path = mmdet_configs_path / "scnet"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/scnet"
)

resnet50_fpn_1x_coco = MMDetscnetBackboneConfig(
    config_path=base_config_path / "scnet_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco-c3f09857.pth",
)

resnet50_fpn_20e_coco = MMDetscnetBackboneConfig(
    config_path=base_config_path / "scnet_r50_fpn_20e_coco.py",
    weights_url=f"{base_weights_url}/scnet_r50_fpn_20e_coco/scnet_r50_fpn_20e_coco-a569f645.pth",
)

resnet101_fpn_20e_coco = MMDetscnetBackboneConfig(
    config_path=base_config_path / "scnet_r101_fpn_20e_coco.py",
    weights_url=f"{base_weights_url}/scnet_r101_fpn_20e_coco/scnet_r101_fpn_20e_coco-294e312c.pth",
)

resnext101_64x4d_fpn_20e_coco = MMDetscnetBackboneConfig(
    config_path=base_config_path / "scnet_x101_64x4d_fpn_20e_coco.py",
    weights_url=f"{base_weights_url}/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth",
)