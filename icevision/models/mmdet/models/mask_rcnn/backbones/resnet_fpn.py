__all__ = [
    "resnet50_caffe_fpn_1x",
    "resnet50_fpn_1x",
    "resnet50_fpn_2x",
    "resnet101_caffe_fpn_1x",
    "resnet101_fpn_1x",
    "resnet101_fpn_2x",
    "resnext101_32x4d_fpn_1x",
    "resnext101_32x4d_fpn_2x",
    "resnext101_64x4d_fpn_1x",
    "resnext101_64x4d_fpn_2x",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MaskRCNNMMDetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="mask_rcnn", **kwargs)


base_config_path = mmdet_configs_path / "mask_rcnn"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn"

resnet50_caffe_fpn_1x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_r50_caffe_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth",
)

resnet50_fpn_1x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
)

resnet50_fpn_2x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_r50_fpn_2x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth",
)

resnet101_caffe_fpn_1x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_r101_caffe_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_r101_caffe_fpn_1x_coco/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth",
)

resnet101_fpn_1x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_r101_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth",
)

resnet101_fpn_2x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_r101_fpn_2x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth",
)

resnext101_32x4d_fpn_1x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_x101_32x4d_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth",
)

resnext101_32x4d_fpn_2x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_x101_32x4d_fpn_2x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth",
)

resnext101_64x4d_fpn_1x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_x101_64x4d_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth",
)

resnext101_64x4d_fpn_2x = MaskRCNNMMDetBackboneConfig(
    config_path=base_config_path / "mask_rcnn_x101_64x4d_fpn_2x_coco.py",
    weights_url=f"{base_weights_url}/mask_rcnn_x101_64x4d_fpn_2x_coco/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth",
)
