__all__ = ["default_prepare_record"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *


def prepare_copy(record):
    return record.copy()


def prepare_img(record):
    if "filepath" in record:
        record["img"] = open_img(record["filepath"])
    return record


def prepare_img_size(record):
    if "img" in record:
        record["height"], record["width"], _ = record["img"].shape
    return record


def prepare_mask(record):
    if "masks" in record:
        record["masks"] = MaskArray.from_masks(
            record["masks"], record["height"], record["width"]
        )
    return record


default_prepare_record = compose(
    prepare_copy,
    prepare_img,
    prepare_img_size,
    prepare_mask,
)
