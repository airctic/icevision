__all__ = [
    "r50_caffe_fpn_1x",
    "r50_fpn_1x",
    "r50_fpn_2x",
    "r101_caffe_fpn_1x",
    "r101_fpn_1x",
    "r101_fpn_2x",
    "x101_32x4d_fpn_1x",
    "x101_32x4d_fpn_2x",
    "x101_64x4d_fpn_1x",
    "x101_64x4d_fpn_2x",
]

from icevision.models.mmdet.utils import *


r50_caffe_fpn_1x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth",
)

r50_fpn_1x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/_base_/models/mask_rcnn_r50_fpn.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
)

r50_fpn_2x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth",
)

r101_caffe_fpn_1x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth",
)

r101_fpn_1x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth",
)

r101_fpn_2x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth",
)

x101_32x4d_fpn_1x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth",
)

x101_32x4d_fpn_2x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth",
)

x101_64x4d_fpn_1x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth",
)

x101_64x4d_fpn_2x = MMDetBackboneConfig(
    model_name="mask_rcnn",
    cfg_filepath="mmdetection/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth",
)
