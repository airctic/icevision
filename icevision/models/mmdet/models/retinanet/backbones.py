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

from icevision.models.mmdet.utils import *

resnet50_caffe_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path
    / "configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth",
)

resnet50_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path / "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
)

resnet50_fpn_2x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path / "configs/retinanet/retinanet_r50_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
)

resnet101_caffe_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path
    / "configs/retinanet/retinanet_r101_caffe_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531-b428fa0f.pth",
)

resnet101_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path / "configs/retinanet/retinanet_r101_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth",
)

resnet101_fpn_2x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path / "configs/retinanet/retinanet_r101_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth",
)

resnext101_32x4d_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path
    / "configs/retinanet/retinanet_x101_32x4d_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth",
)

resnext101_32x4d_fpn_2x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path
    / "configs/retinanet/retinanet_x101_32x4d_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth",
)

resnext101_64x4d_fpn_1x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path
    / "configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth",
)

resnext101_64x4d_fpn_2x = MMDetBackboneConfig(
    model_name="retinanet",
    config_path=mmdet_configs_path
    / "configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py",
    weights_url="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth",
)
