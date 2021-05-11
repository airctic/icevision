__all__ = [
    "convert_background_from_zero_to_last",
    "convert_background_from_last_to_zero",
    "mmdet_tensor_to_image",
    "build_model",
]

from icevision.imports import *
from icevision.utils import *
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from icevision.models.mmdet.utils import *


def convert_background_from_zero_to_last(label_ids, class_map):
    label_ids = label_ids - 1
    label_ids[label_ids == -1] = class_map.num_classes - 1
    return label_ids


def convert_background_from_last_to_zero(label_ids, class_map):
    label_ids = label_ids + 1
    label_ids[label_ids == len(class_map)] = 0
    return label_ids


def mmdet_tensor_to_image(tensor_image):
    return tensor_to_image(tensor_image)[:, :, ::-1].copy()


def build_model(
    model_type: str,
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

    if model_type == "one_stage_detector_bbox":
        cfg.model.bbox_head.num_classes = num_classes - 1

    if model_type == "two_stage_detector_bbox":
        # Sparse-RCNN has a list of bbox_head whereas Faster-RCNN has only one
        if isinstance(cfg.model.roi_head.bbox_head, list):
            for bbox_head in cfg.model.roi_head.bbox_head:
                bbox_head["num_classes"] = num_classes - 1
        else:
            cfg.model.roi_head.bbox_head.num_classes = num_classes - 1

    if model_type == "two_stage_detector_mask":
        cfg.model.roi_head.bbox_head.num_classes = num_classes - 1
        cfg.model.roi_head.mask_head.num_classes = num_classes - 1

    if (pretrained == False) or (weights_path is not None):
        cfg.model.pretrained = None

    _model = build_detector(cfg.model, cfg.get("train_cfg"), cfg.get("test_cfg"))
    _model.init_weights()

    if pretrained and (weights_path is not None):
        load_checkpoint(_model, str(weights_path))

    _model.param_groups = MethodType(param_groups, _model)

    return _model
