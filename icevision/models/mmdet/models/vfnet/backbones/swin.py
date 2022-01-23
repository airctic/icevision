__all__ = [
    "swin_t_p4_w7_fpn_1x_coco",
    "swin_s_p4_w7_fpn_1x_coco",
    "swin_b_p4_w7_fpn_1x_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *
from urllib.request import urlopen


class MMDetSwinBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="swin", **kwargs)


base_config_path = mmdet_configs_path / "swin"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/swin"
)

#####
# Download congi from URL.
config_url = "https://raw.githubusercontent.com/dnth/mmdetection_configs/custom-configs/configs/swin/vfnet_swin_tiny_patch4_window7_224.py"
with urlopen(config_url) as webpage:
    content = webpage.read()
# Save to local mmdet config folder.
with open(base_config_path / "vfnet_swin_tiny_patch4_window7_224.py", "wb") as download:
    download.write(content)


config_url = "https://raw.githubusercontent.com/dnth/mmdetection_configs/custom-configs/configs/swin/vfnet_swin_small_patch4_window7_224.py"
with urlopen(config_url) as webpage:
    content = webpage.read()
# Save to local mmdet config folder.
with open(
    base_config_path / "vfnet_swin_small_patch4_window7_224.py", "wb"
) as download:
    download.write(content)


config_url = "https://raw.githubusercontent.com/dnth/mmdetection_configs/custom-configs/configs/swin/vfnet_swin_base_patch4_window7_224.py"
with urlopen(config_url) as webpage:
    content = webpage.read()
# Save to local mmdet config folder.
with open(
    base_config_path / "vfnet_base_small_patch4_window7_224.py", "wb"
) as download:
    download.write(content)

swin_t_p4_w7_fpn_1x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path
    / "vfnet_swin_tiny_patch4_window7_224.py",  # Need to host this online on github
    weights_url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
)

swin_s_p4_w7_fpn_1x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path
    / "vfnet_swin_small_patch4_window7_224.py",  # Need to host this online on github
    weights_url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
)

swin_b_p4_w7_fpn_1x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path
    / "vfnet_swin_base_patch4_window7_224.py",  # Need to host this online on github
    weights_url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
)
