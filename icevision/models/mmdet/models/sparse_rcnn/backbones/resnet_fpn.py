__all__ = [
    "resnet50_fpn_1x",
    "resnet50_fpn_mstrain_480_800_3x",
    "resnet50_fpn_300_proposals_crop_mstrain_480_800_3x",
    "resnet101_fpn_mstrain_480_800_3x_coco",
    "resnet101_fpn_300_proposals_crop_mstrain_480_800_3x",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetSparseRCNNBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="sparse_rcnn", **kwargs)


base_config_path = mmdet_configs_path / "sparse_rcnn"
base_weights_url = "https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn"

resnet50_fpn_1x = MMDetSparseRCNNBackboneConfig(
    config_path=base_config_path / "sparse_rcnn_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth",
)

resnet50_fpn_mstrain_480_800_3x = MMDetSparseRCNNBackboneConfig(
    config_path=base_config_path / "sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco.py",
    weights_url=f"{base_weights_url}/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth",
)

resnet50_fpn_300_proposals_crop_mstrain_480_800_3x = MMDetSparseRCNNBackboneConfig(
    config_path=base_config_path
    / "sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py",
    weights_url=f"{base_weights_url}/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth",
)

resnet101_fpn_mstrain_480_800_3x_coco = MMDetSparseRCNNBackboneConfig(
    config_path=base_config_path / "sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco.py",
    weights_url=f"{base_weights_url}/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth",
)

resnet101_fpn_300_proposals_crop_mstrain_480_800_3x = MMDetSparseRCNNBackboneConfig(
    config_path=base_config_path
    / "sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py",
    weights_url=f"{base_weights_url}/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth",
)
