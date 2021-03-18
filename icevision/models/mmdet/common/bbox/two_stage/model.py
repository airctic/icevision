__all__ = ["model"]

from icevision.imports import *
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from icevision.models.mmdet.utils import *


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

    # Sparse-RCNN has a list of bbox_head whereas Faster-RCNN has only one
    if isinstance(cfg.model.roi_head.bbox_head, list):
        for bbox_head in cfg.model.roi_head.bbox_head:
            bbox_head["num_classes"] = num_classes - 1
    else:
        cfg.model.roi_head.bbox_head.num_classes = num_classes - 1

    if weights_path is not None:
        cfg.model.pretrained = None

    _model = build_detector(cfg.model, cfg.get("train_cfg"), cfg.get("test_cfg"))

    if weights_path is not None:
        load_checkpoint(_model, str(weights_path))

    _model.param_groups = MethodType(param_groups, _model)

    return _model
