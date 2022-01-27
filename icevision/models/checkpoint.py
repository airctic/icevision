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


# COCO Classes: 80 classes
CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def save_icevision_checkpoint(
    model,
    filename,
    optimizer=None,
    meta=None,
    model_name=None,
    backbone_name=None,
    classes=None,
    img_size=None,
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
    model_name=None,
    backbone_name=None,
    classes=None,
    is_coco=False,
    img_size=None,
    map_location=None,
    strict=False,
    revise_keys=[
        (r"^module\.", ""),
    ],
    eval_mode=True,
    logger=None,
):
    """load checkpoint through URL scheme path.
    Args:
        filename (str): checkpoint file name with given prefix
        map_location (str, optional): Same as :func:`torch.load`.
            Default: None
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    if isinstance(filename, Path):
        filename = str(filename)

    checkpoint = _load_checkpoint(
        filename=filename, map_location=map_location, logger=logger
    )

    if is_coco and classes:
        err_msg = "`is_coco` cannot be set to True if `classes` is passed and `not None`. `classes` has priority. `is_coco` will be ignored."
        if logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

    if classes is None:
        if is_coco:
            classes = CLASSES
        else:
            classes = checkpoint["meta"].get("classes", None)

    class_map = None
    if classes:
        class_map = ClassMap(classes)
        num_classes = len(class_map)

    if img_size is None:
        img_size = checkpoint["meta"].get("img_size", None)

    if model_name is None:
        model_name = checkpoint["meta"].get("model_name", None)

    model_type = None
    if model_name:
        model = model_name.split(".")
        # If model_name contains three or more components, the library and model are the second to last and last components, respectively (e.g. models.mmdet.retinanet)
        if len(model) >= 3:
            model_type = getattr(getattr(models, model[-2]), model[-1])
        # If model_name follows the default convention, the library and model are the first and second components, respectively (e.g. mmdet.retinanet)
        else:
            model_type = getattr(getattr(models, model[0]), model[1])

    if backbone_name is None:
        backbone_name = checkpoint["meta"].get("backbone_name", None)
    if model_type and backbone_name:
        backbone = getattr(model_type.backbones, backbone_name)

    extra_args = {}
    if img_size is None:
        img_size = checkpoint["meta"].get("img_size", None)

    models_with_img_size = ("yolov5", "efficientdet")
    # if 'efficientdet' in model_name:
    if (model_name) and (any(m in model_name for m in models_with_img_size)):
        extra_args["img_size"] = img_size

    # Instantiate model
    if model_type and backbone:
        model = model_type.model(
            backbone=backbone(pretrained=False), num_classes=num_classes, **extra_args
        )
    else:
        model = None

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
    if model:
        load_state_dict(model, state_dict, strict, logger)
    if eval_mode:
        model.eval()

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
