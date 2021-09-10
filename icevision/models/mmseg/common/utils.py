__all__ = [
    "convert_background_from_zero_to_last",
    "convert_background_from_last_to_zero",
    "mmseg_tensor_to_image",
    "build_model",
]

from icevision.imports import *
from icevision.utils import *
from mmcv import Config
from mmseg.apis import init_segmentor
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from icevision.models.mmseg.utils import *
from icevision.core.exceptions import *


def convert_background_from_zero_to_last(label_ids, class_map):
    label_ids = label_ids - 1
    label_ids[label_ids == -1] = class_map.num_classes - 1
    return label_ids


def convert_background_from_last_to_zero(label_ids, class_map):
    label_ids = label_ids + 1
    label_ids[label_ids == len(class_map)] = 0
    return label_ids


def mmseg_tensor_to_image(tensor_image):
    return tensor_to_image(tensor_image)[:, :, ::-1].copy()


def build_model(
    model_type: str,
    backbone: MMSegBackboneConfig,
    num_classes: int,
    pretrained: bool = True,
    checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
    force_download=False,
    cfg_options=None,
) -> nn.Module:

    cfg, weights_path = create_model_config(
        backbone=backbone,
        pretrained=pretrained,
        checkpoints_path=checkpoints_path,
        force_download=force_download,
        cfg_options=cfg_options,
    )

    if model_type == "EncoderDecoder":
        cfg.model.decode_head.num_classes = num_classes - 1
    else:
        raise (InvalidMMSegModelType)

    if (pretrained == False) or (weights_path is not None):
        cfg.model.pretrained = None
        _model = build_segmentor(cfg.model, cfg.get("train_cfg"), cfg.get("test_cfg"))
        # _model.init_weights()
    else:
        print(weights_path)
        _model = init_segmentor(cfg.model, weights_path)
        # _model.init_weights()
        # load_checkpoint(_model, str(weights_path))

    # _model.param_groups = MethodType(param_groups, _model) #TODO Re-enable this
    _model.param_groups = []
    _model.cfg = cfg  # save the config in the model for convenience
    _model.weights_path = weights_path  # save the model.weights_path in case we want to rebuild the model after updating its attributes

    return _model
