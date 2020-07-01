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
        data["img"],
        labels=data["labels"] if (label and "labels" in record) else None,
        bboxes=data["bboxes"] if (bbox and "bboxes" in record) else None,
        masks=data["masks"] if (mask and "masks" in record) else None,
        ax=ax,
        show=show,
    )
