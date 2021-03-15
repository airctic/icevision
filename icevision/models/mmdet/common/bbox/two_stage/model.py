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

    # if `cfg` argument is a path (str, Path) create an Config object from the file
    # otherwise cfg should be already an Config object
    if isinstance(cfg, (str, Path)):
        cfg = Config.fromfile(str(cfg))

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

    return _model
