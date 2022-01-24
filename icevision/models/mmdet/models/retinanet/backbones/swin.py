__all__ = [
    "swin_t_p4_w7_fpn_1x_coco",
    "swin_s_p4_w7_fpn_1x_coco",
    "swin_b_p4_w7_fpn_1x_coco",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *
from urllib.request import urlopen, urlcleanup


class MMDetSwinBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="swin", **kwargs)


base_config_path = mmdet_configs_path / "swin"
base_weights_url = (
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/swin"
)

######## Download swin configs for RetinaNet ################

config_urls = [
    "https://raw.githubusercontent.com/dnth/mmdetection_configs/custom-configs/configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py",
    "https://raw.githubusercontent.com/dnth/mmdetection_configs/custom-configs/configs/swin/retinanet_swin-s-p4-w7_fpn_1x_coco.py",
    "https://raw.githubusercontent.com/dnth/mmdetection_configs/custom-configs/configs/swin/retinanet_swin-b-p4-w7_fpn_1x_coco.py",
]

for url in config_urls:
    urlcleanup()  # Clear url cache
    with urlopen(url) as webpage:
        content = webpage.read()
    with open(base_config_path / url.split("/")[-1], "wb") as download:
        download.write(content)

########################################################

swin_t_p4_w7_fpn_1x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path / "retinanet_swin-t-p4-w7_fpn_1x_coco.py",
    weights_url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
)

swin_s_p4_w7_fpn_1x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path / "retinanet_swin-s-p4-w7_fpn_1x_coco.py",
    weights_url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
)

swin_b_p4_w7_fpn_1x_coco = MMDetSwinBackboneConfig(
    config_path=base_config_path / "retinanet_swin-b-p4-w7_fpn_1x_coco.py",
    weights_url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
)
