__all__ = ["hourglass104_mstest_16x6_210e_coco"]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetCentripetalnetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="centripetalnet", **kwargs)


base_config_path = mmdet_configs_path / "centripetalnet"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/centripetalnet"

hourglass104_mstest_16x6_210e_coco = MMDetCentripetalnetBackboneConfig(
    config_path=base_config_path
    / "centripetalnet_hourglass104_mstest_16x6_210e_coco.py",
    weights_url=f"{base_weights_url}/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth",
)
