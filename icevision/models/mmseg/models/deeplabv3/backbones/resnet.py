__all__ = [
    "resnet101_d8",
]

from icevision.imports import *
from icevision.models.mmseg.utils import *


class MMSegDeeplabBackboneConfig(MMSegBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="deeplabv3", **kwargs)


base_config_path = mmseg_configs_path / "deeplabv3"
base_weights_url = "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3"

resnet101_d8 = MMSegDeeplabBackboneConfig(
    backbone_type="R-101-D8",
    pre_trained_variants=[
        {
            "pre_training_dataset": "pascal_context",
            "weights_url": f"{base_weights_url}/deeplabv3_r101-d8_480x480_40k_pascal_context/deeplabv3_r101-d8_480x480_40k_pascal_context_20200911_204118-1aa27336.pth",
            "config_path": base_config_path
            / "deeplabv3_r50-d8_480x480_40k_pascal_context.py",
            "crop_size": (480, 480),
            "lr_schd": 40000,
            "default": True,
        },
        {
            "pre_training_dataset": "pascal_context",
            "weights_url": "'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context/deeplabv3_r101-d8_480x480_80k_pascal_context_20200911_170155-2a21fff3.pth",
            "config_path": base_config_path
            / "deeplabv3_r101-d8_480x480_80k_pascal_context.py",
            "crop_size": (480, 480),
            "lr_schd": 80000,
        },
        {
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x1024_40k_cityscapes/deeplabv3_r101-d8_512x1024_40k_cityscapes_20200605_012241-7fd3f799.pth",
            "config_path": base_config_path
            / "deeplabv3_r101-d8_512x1024_40k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": 40000,
        },
    ],
)
