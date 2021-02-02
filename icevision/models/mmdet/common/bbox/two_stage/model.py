__all__ = ["model"]

from icevision.imports import *
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint


def model(
    cfg_path: Union[str, Path],
    num_classes: int,
    weights_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    cfg = Config.fromfile(str(cfg_path))
    cfg.model.roi_head.bbox_head.num_classes = num_classes - 1
    if weights_path is not None:
        cfg.model.pretrained = None

    _model = build_detector(cfg.model, cfg.train_cfg, cfg.test_cfg)

    if weights_path is not None:
        load_checkpoint(_model, str(weights_path))

    return _model
