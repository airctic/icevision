__all__ = [
    "r50_16x2_50e_coco",
    "refine_r50_16x2_50e_coco",
    "twostage_refine_r50_16x2_50e_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetDeformableDetrBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="deformable_detr", **kwargs)


base_config_path = mmdet_configs_path / "deformable_detr"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/deformable_detr"

r50_16x2_50e_coco = MMDetDeformableDetrBackboneConfig(
    config_path=base_config_path / "deformable_detr_r50_16x2_50e_coco.py",
    weights_url=f"{base_weights_url}/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth",
)

refine_r50_16x2_50e_coco = MMDetDeformableDetrBackboneConfig(
    config_path=base_config_path / "deformable_detr_refine_r50_16x2_50e_coco.py",
    weights_url=f"{base_weights_url}/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth",
)

twostage_refine_r50_16x2_50e_coco = (
    refine_r50_16x2_50e_coco
) = MMDetDeformableDetrBackboneConfig(
    config_path=base_config_path
    / "deformable_detr_twostage_refine_r50_16x2_50e_coco.py",
    weights_url=f"{base_weights_url}/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth",
)
