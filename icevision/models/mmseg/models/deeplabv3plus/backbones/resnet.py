__all__ = [
    "resnet101_d8",
]

from icevision.imports import *
from icevision.models.mmseg.utils import *


class MMSegDeeplabBackboneConfig(MMSegBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="deeplabv3plus", **kwargs)


base_config_path = mmseg_configs_path / "deeplabv3plus"
base_weights_url = "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus"

resnet101_d8 = MMSegDeeplabBackboneConfig(
    backbone_type="R-101-D8",
    pre_trained_variants=[
        {
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_769x769_40k_cityscapes/deeplabv3plus_r101-d8_769x769_40k_cityscapes_20200606_114304-ff414b9e.pth",
            "config_path": base_config_path
            / "deeplabv3plus_r101-d8_512x1024_40k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": 40000,
            "default": True,
        },
    ],
)
