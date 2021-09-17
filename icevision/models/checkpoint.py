__all__ = ["save_icevision_checkpoint", "model_from_checkpoint"]

from icevision.imports import *
from icevision.core import *
from icevision import models

from mmcv.runner import (
    load_checkpoint,
    save_checkpoint,
    _load_checkpoint,
    load_state_dict,
)


def save_icevision_checkpoint(
    model,
    model_name,
    backbone_name,
    class_map,
    img_size,
    filename,
    optimizer=None,
    meta=None,
):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.

    Examples:
        >>> save_icevision_checkpoint(model_saved,
                        model_name='mmdet.retinanet',
                        backbone_name='resnet50_fpn_1x',
                        class_map =  class_map,
                        img_size=img_size,
                        filename=checkpoint_path,
                        meta={'icevision_version': '0.9.1'})

    """

    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f"meta must be a dict or None, but got {type(meta)}")

    if class_map:
        classes = class_map._id2class
    else:
        classes = None

    if classes:
        meta.update(classes=classes)

    if model_name:
        meta.update(model_name=model_name)

    if img_size:
        meta.update(img_size=img_size)

    if backbone_name:
        meta.update(backbone_name=backbone_name)

    save_checkpoint(model, filename, optimizer=optimizer, meta=meta)


def model_from_checkpoint(
    filename: Union[Path, str],
    map_location=None,
    strict=False,
    logger=None,
    revise_keys=[(r"^module\.", "")],
):
    """load checkpoint through URL scheme path.

    Args:
        filename (str): checkpoint file name with given prefix
        map_location (str, optional): Same as :func:`torch.load`.
            Default: None
        logger (:mod:`logging.Logger`, optional): The logger for message.
            Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    if isinstance(filename, Path):
        filename = str(filename)

    checkpoint = _load_checkpoint(filename)

    class_map = None
    num_classes = None
    img_size = None
    model_name = None
    backbone = None

    classes = checkpoint["meta"].get("classes", None)
    if classes:
        class_map = ClassMap(checkpoint["meta"]["classes"])
        num_classes = len(class_map)

    img_size = checkpoint["meta"].get("img_size", None)

    model_name = checkpoint["meta"].get("model_name", None)
    if model_name:
        lib, mod = model_name.split(".")
        model_type = getattr(getattr(models, lib), mod)

    backbone_name = checkpoint["meta"].get("backbone_name", None)
    if backbone_name:
        backbone = getattr(model_type.backbones, backbone_name)

    extra_args = {}
    models_with_img_size = ("yolov5", "efficientdet")
    # if 'efficientdet' in model_name:
    if any(m in model_name for m in models_with_img_size):
        extra_args["img_size"] = img_size

    # Instantiate model
    model = model_type.model(
        backbone=backbone(pretrained=False), num_classes=num_classes, **extra_args
    )

    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)

    checkpoint_and_model = {
        "model": model,
        "model_type": model_type,
        "backbone": backbone,
        "class_map": class_map,
        "img_size": img_size,
        "checkpoint": checkpoint,
    }
    return checkpoint_and_model
    # return model, model_type, backbone, class_map, img_size, checkpoint
