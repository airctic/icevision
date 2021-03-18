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
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth",
)

r50_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/_base_/models/retinanet_r50_fpn.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
)

r50_fpn_2x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
)

r101_caffe_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_r101_caffe_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531-b428fa0f.pth",
)

r101_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_r101_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth",
)

r101_fpn_2x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_r101_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth",
)

x101_32x4d_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_x101_32x4d_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth",
)

x101_32x4d_fpn_2x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_x101_32x4d_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth",
)

x101_64x4d_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth",
)

x101_64x4d_fpn_2x = MMDetBackboneConfig(
    model_name="retinanet",
    cfg_filepath="mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth",
)
