__all__ = ["model"]

from icevision.imports import *
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from icevision.models.mmdet.utils import *
from icevision.utils.download_utils import *


def model(
    backbone,
    num_classes: int,
    pretrained: bool = True,
    checkpoints_path: Optional[Union[str, Path]] = None,
    force_download=False,
) -> nn.Module:

    save_dir = None

    model_name = backbone["model_name"]
    cfg_filepath = backbone["cfg_filepath"]
    weights_url = backbone["weights_url"]

    # download weights
    if pretrained and weights_url:
        if checkpoints_path:
            save_dir = checkpoints_path.model_name
        else:
            save_dir = Path("checkpoints") / model_name
        save_dir.mkdir(exist_ok=True, parents=True)

        fname = Path(weights_url).name
        weights_path = save_dir / fname

        if not weights_path.exists() or force_download:
            download_url(weights_url, weights_path)

    if cfg_filepath:
        cfg = Config.fromfile(cfg_filepath)
    else:
        raise RuntimeError("model must have a valid configuration file")

    cfg.model.bbox_head.num_classes = num_classes - 1
    if pretrained and (weights_path is not None):
        cfg.model.pretrained = None

    _model = build_detector(cfg.model, cfg.get("train_cfg"), cfg.get("test_cfg"))

    if pretrained and (weights_path is not None):
        load_checkpoint(_model, str(weights_path))

    _model.param_groups = MethodType(param_groups, _model)

    return _model
