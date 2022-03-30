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
from mmcv.cnn.utils import revert_sync_batchnorm


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
    pre_training_dataset: str = None,
    lr_schd: str = None,
    crop_size: tuple = None,
    logical_param_groups: bool = False,
    device: str = "gpu",
    distributed: bool = False,
) -> nn.Module:

    if pre_training_dataset is None or lr_schd is None or crop_size is None:
        pre_trained_variant = backbone.get_default_pre_trained_variant()
        logger.info(
            f"Loaded default configuration for {backbone.model_name} with {backbone.backbone_type} backbone (Pre-training dataset = {pre_trained_variant['pre_training_dataset']}, learning schedule = {pre_trained_variant['lr_schd']}, crop size = {pre_trained_variant['crop_size']})"
        )
    else:
        pre_trained_variant = backbone.get_pre_trained_variant(
            pre_training_dataset, lr_schd, crop_size
        )
        logger.info(
            f"Loaded non-default configuration for {backbone.model_name} with {backbone.backbone_type} backbone (Pre-training dataset = {pre_trained_variant['pre_training_dataset']}, learning schedule = {pre_trained_variant['lr_schd']}, crop size = {pre_trained_variant['crop_size']})"
        )

    cfg, weights_path = create_model_config(
        model_name=backbone.model_name,
        weights_url=pre_trained_variant["weights_url"],
        config_path=pre_trained_variant["config_path"],
        pretrained=pretrained,
        checkpoints_path=checkpoints_path,
        force_download=force_download,
        cfg_options=cfg_options,
    )

    if model_type == "EncoderDecoder":
        if "auxiliary_head" in cfg.model:
            cfg.model.auxiliary_head.num_classes = num_classes

        if "decode_head" in cfg.model:
            cfg.model.decode_head.num_classes = num_classes
    else:
        raise (InvalidMMSegModelType)

    _model = build_segmentor(cfg.model, cfg.get("train_cfg"), cfg.get("test_cfg"))

    if _model.backbone.init_cfg:
        _model.backbone.init_cfg = {}

    _model.init_weights()

    if pretrained == True:
        load_checkpoint(_model, str(weights_path))

    if logical_param_groups:
        _model.param_groups = MethodType(param_groups, _model)
    else:
        _model.param_groups = MethodType(param_groups_default, _model)

    # TODO: Implement support for distributed treaning
    if not distributed:
        _model = revert_sync_batchnorm(_model)

    _model.cfg = cfg  # save the config in the model for convenience
    _model.weights_path = weights_path  # save the model.weights_path in case we want to rebuild the model after updating its attributes

    return _model
