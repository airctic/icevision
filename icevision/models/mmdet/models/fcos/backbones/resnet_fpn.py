__all__ = [
    "resnet50_caffe_fpn_gn_head",
    "resnet50_caffe_fpn_gn_head_center_normbbox_centeronreg_giou",
    "resnet50_caffe_fpn_gn_head_dcn_1x_center_normbbox_centeronreg_giou",
    "resnet101_caffe_fpn_gn_head_1x_coco",
    "resnet50_caffe_fpn_gn_head_mstrain_640_800_2x",
    "resnet101_caffe_fpn_gn_head_mstrain_640_800_2x",
    "resnext101_64x4d_fpn_gn_head_mstrain_640_800_2x",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetFCOSBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="fcos", **kwargs)


base_config_path = mmdet_configs_path / "fcos"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos"
)

resnet50_caffe_fpn_gn_head = MMDetFCOSBackboneConfig(
    config_path=base_config_path / "fcos_r50_caffe_fpn_gn-head_1x_coco.py",
    weights_url=f"{base_weights_url}/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth",
)

resnet50_caffe_fpn_gn_head_center_normbbox_centeronreg_giou = MMDetFCOSBackboneConfig(
    config_path=base_config_path
    / "fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py",
    weights_url=f"{base_weights_url}/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth",
)

resnet50_caffe_fpn_gn_head_dcn_1x_center_normbbox_centeronreg_giou = MMDetFCOSBackboneConfig(
    config_path=base_config_path
    / "fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco.py",
    weights_url=f"{base_weights_url}/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth",
)

resnet101_caffe_fpn_gn_head_1x_coco = MMDetFCOSBackboneConfig(
    config_path=base_config_path / "fcos_r101_caffe_fpn_gn-head_1x_coco.py",
    weights_url=f"{base_weights_url}/fcos_r101_caffe_fpn_gn-head_1x_coco/fcos_r101_caffe_fpn_gn-head_1x_coco-0e37b982.pth",
)

resnet50_caffe_fpn_gn_head_mstrain_640_800_2x = MMDetFCOSBackboneConfig(
    config_path=base_config_path
    / "fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py",
    weights_url=f"{base_weights_url}/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth",
)

resnet101_caffe_fpn_gn_head_mstrain_640_800_2x = MMDetFCOSBackboneConfig(
    config_path=base_config_path
    / "fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py",
    weights_url=f"{base_weights_url}/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth",
)

resnext101_64x4d_fpn_gn_head_mstrain_640_800_2x = MMDetFCOSBackboneConfig(
    config_path=base_config_path
    / "fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py",
    weights_url=f"{base_weights_url}/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth",
)
