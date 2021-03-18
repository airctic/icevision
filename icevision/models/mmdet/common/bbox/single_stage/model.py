__all__ = ["model"]

from icevision.imports import *
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from icevision.models.mmdet.utils import *
from icevision.utils.download_utils import *


def model(
    backbone: MMDetBackboneConfig,
    num_classes: int,
    pretrained: bool = True,
    checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
    force_download=False,
) -> nn.Module:

    cfg, weights_path = create_model_config(
        backbone=backbone,
        pretrained=pretrained,
        checkpoints_path=checkpoints_path,
        force_download=force_download,
    )

    cfg.model.bbox_head.num_classes = num_classes - 1

    if pretrained and (weights_path is not None):
        cfg.model.pretrained = None

    _model = build_detector(cfg.model, cfg.get("train_cfg"), cfg.get("test_cfg"))

    if pretrained and (weights_path is not None):
        load_checkpoint(_model, str(weights_path))

    _model.param_groups = MethodType(param_groups, _model)

    return _model
