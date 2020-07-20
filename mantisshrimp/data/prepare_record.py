__all__ = ["default_prepare_record"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *


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
    prepare_copy, prepare_img, prepare_mask, prepare_img_size
)
