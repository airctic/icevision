__all__ = ["swin_t_p4_w7_fpn_1x_coco"]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetSwinBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="swin", **kwargs)


base_config_path = mmdet_configs_path / "swin"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/swin"
)


# Download the RetinaNet-Swin config from the mmdet dev branch (the config from the current master branch does not work)
# https://github.com/open-mmlab/mmdetection/pull/6973
# This might not be needed in the future when the Swin config is added into the master branch
from urllib.request import urlopen

# Download from URL.
config_url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/dev/configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py"
with urlopen(config_url) as webpage:
    content = webpage.read()
# Save to local mmdet config folder.
with open(base_config_path / "retinanet_swin-t-p4-w7_fpn_1x_coco.py", "wb") as download:
    download.write(content)


swin_t_p4_w7_fpn_1x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path / "retinanet_swin-t-p4-w7_fpn_1x_coco.py",
    weights_url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
)
