__all__ = [
    "mask_rcnn_swin_t_p4_w7_fpn_1x_coco",
    "mask_rcnn_swin_t_p4_w7_fpn_ms_crop_3x_coco",
    "mask_rcnn_swin_t_p4_w7_fpn_fp16_ms_crop_3x_coco",
    "mask_rcnn_swin_s_p4_w7_fpn_fp16_ms_crop_3x_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetSwinBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="swin", **kwargs)


base_config_path = mmdet_configs_path / "swin"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/swin"
)


mask_rcnn_swin_t_p4_w7_fpn_1x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path / "mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
)

mask_rcnn_swin_t_p4_w7_fpn_ms_crop_3x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path / "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth",
)

mask_rcnn_swin_t_p4_w7_fpn_fp16_ms_crop_3x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path / "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth",
)

mask_rcnn_swin_s_p4_w7_fpn_fp16_ms_crop_3x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path / "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth",
)
