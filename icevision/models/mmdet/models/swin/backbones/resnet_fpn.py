__all__ = ["swin_t_1x"]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetSwinBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="swin", **kwargs)


base_config_path = mmdet_configs_path / "swin"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/swin"
)


swin_t_1x = MMDetSwinBackboneConfig(
    config_path=base_config_path / "mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
)
