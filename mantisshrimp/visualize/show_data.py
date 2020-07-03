__all__ = ["show_sample", "show_record"]

from mantisshrimp.imports import *
from mantisshrimp.data import *
from mantisshrimp.visualize.show_annotation import *


def show_sample(
    sample,
    denormalize_fn=None,
    label=True,
    bbox=True,
    mask=True,
    show=False,
    ax: plt.Axes = None,
):
    return show_annotation(
        img=sample["img"] if denormalize_fn is None else denormalize_fn(sample["img"]),
        labels=sample["labels"] if (label and "labels" in sample) else None,
        bboxes=sample["bboxes"] if (bbox and "bboxes" in sample) else None,
        masks=sample["masks"] if (mask and "masks" in sample) else None,
        ax=ax,
        show=show,
    )


def show_record(
    record,
    label: bool = True,
    bbox: bool = True,
    mask: bool = True,
    ax: plt.Axes = None,
    show: bool = False,
    prepare_record=None,
):
    data_preparer = prepare_record or default_prepare_record
    sample = data_preparer(record)
    return show_sample(
        sample=sample, label=label, bbox=bbox, mask=mask, ax=ax, show=show
    )
