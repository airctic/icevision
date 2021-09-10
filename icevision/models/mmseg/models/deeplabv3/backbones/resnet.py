__all__ = [
    "resnet50_d8",
]

from icevision.imports import *
from icevision.models.mmseg.utils import *


class MMSegDeeplabBackboneConfig(MMSegBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="deeplabv3", **kwargs)


base_config_path = mmseg_configs_path / "deeplabv3"
base_weights_url = "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3"

resnet50_d8 = MMSegDeeplabBackboneConfig(
    config_path=base_config_path / "deeplabv3_r50-d8_480x480_40k_pascal_context.py",
    weights_url=f"{base_weights_url}/deeplabv3_r101-d8_480x480_40k_pascal_context/deeplabv3_r101-d8_480x480_40k_pascal_context_20200911_204118-1aa27336.pth",
)