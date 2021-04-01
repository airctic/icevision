__all__ = ["model"]

from icevision.imports import *
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from icevision.models.mmdet.utils import *
from icevision.models.mmdet.common.utils import *


def model(
    backbone: MMDetBackboneConfig,
    num_classes: int,
    checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
    force_download=False,
) -> nn.Module:

    return build_model(
        model_type="two_stage_detector_mask",
        backbone=backbone,
        num_classes=num_classes,
        pretrained=backbone.pretrained,
        checkpoints_path=checkpoints_path,
        force_download=force_download,
    )
