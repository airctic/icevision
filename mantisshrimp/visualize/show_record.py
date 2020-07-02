__all__ = ["show_record"]

from mantisshrimp.imports import *
from mantisshrimp.data import *
from mantisshrimp.visualize.show_annotation import *


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
    data = data_preparer(record)
    return show_annotation(
        img=data["img"],
        labels=data["label"] if (label and "label" in record) else None,
        bboxes=data["bbox"] if (bbox and "bbox" in record) else None,
        masks=data["mask"] if (mask and "mask" in record) else None,
        ax=ax,
        show=show,
    )
