__all__ = ["model"]

from icevision.imports import *
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint


def model(
    cfg: Union[str, Path, Config],
    num_classes: int,
    weights_path: Optional[Union[str, Path]] = None,
) -> nn.Module:

    # if `cfg` argument a path (str, Path) create a Config object from the file
    # otherwise cfg is already a Config object
    if isinstance(cfg, (str, Path)):
        cfg = Config.fromfile(str(cfg))

    cfg.model.bbox_head.num_classes = num_classes - 1
    if weights_path is not None:
        cfg.model.pretrained = None

    _model = build_detector(cfg.model, cfg.get("train_cfg"), cfg.get("test_cfg"))

    if weights_path is not None:
        load_checkpoint(_model, str(weights_path))

    return _model
