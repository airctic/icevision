__all__ = [
    "resnet50_fpn_1x",
    "resnet50_fpn_mstrain_2x",
    "resnet50_fpn_mdconv_c3_c5_mstrain_2x",
    "resnet101_fpn_1x",
    "resnet101_fpn_mstrain_2x",
    "resnet101_fpn_mdconv_c3_c5_mstrain_2x",
    "resnext101_32x4d_fpn_mdconv_c3_c5_mstrain_2x",
    "resnext101_64x4d_fpn_mdconv_c3_c5_mstrain_2x",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetVFNETBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="vfnet", **kwargs)


base_config_path = mmdet_configs_path / "vfnet"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet"
)

resnet50_fpn_1x = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth",
)
resnet50_fpn_mstrain_2x = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_r50_fpn_mstrain_2x_coco.py",
    weights_url=f"{base_weights_url}/vfnet_r50_fpn_mstrain_2x_coco/vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth",
)

resnet50_fpn_mdconv_c3_c5_mstrain_2x = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_r50_fpn_mdconv_c3_c5_mstrain_2x.py",
    weights_url=f"{base_weights_url}/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth",
)

resnet101_fpn_1x = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_r101_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/vfnet_r101_fpn_1x_coco/vfnet_r101_fpn_1x_coco_20201027pth-c831ece7.pth",
)

resnet101_fpn_mstrain_2x = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_r101_fpn_mstrain_2x_coco.py",
    weights_url=f"{base_weights_url}/vfnet_r101_fpn_mstrain_2x_coco/vfnet_r101_fpn_mstrain_2x_coco_20201027pth-4a5d53f1.pth",
)

resnet101_fpn_mdconv_c3_c5_mstrain_2x = MMDetVFNETBackboneConfig(
    config_path=base_config_path / "vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
    weights_url=f"{base_weights_url}/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth",
)

resnext101_32x4d_fpn_mdconv_c3_c5_mstrain_2x = MMDetVFNETBackboneConfig(
    config_path=base_config_path
    / "vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
    weights_url=f"{base_weights_url}/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth",
)

resnext101_64x4d_fpn_mdconv_c3_c5_mstrain_2x = MMDetVFNETBackboneConfig(
    config_path=base_config_path
    / "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
    weights_url=f"{base_weights_url}/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth",
)
