__all__ = ["show_record"]

from mantisshrimp.imports import *
from mantisshrimp.data_preparer import *
from .show_annotation import *


def show_record(
    record,
    label: bool = True,
    bbox: bool = True,
    mask: bool = True,
    ax: plt.Axes = None,
    show: bool = False,
    data_preparer: DataPreparer = None,
):
    data_preparer = data_preparer or DefaultDataPreparer()
    data = data_preparer(record)
    return show_annotation(
        data["img"],
        labels=data["label"] if label else None,
        bboxes=data["bbox"] if bbox else None,
        masks=data["mask"] if mask else None,
        ax=ax,
        show=show,
    )
