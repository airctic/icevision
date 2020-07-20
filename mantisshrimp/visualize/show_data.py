__all__ = [
    "show_sample",
    "show_record",
    "show_pred",
    "show_samples",
    "show_records",
    "show_preds",
]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *
from mantisshrimp.parsers import *
from mantisshrimp.data import *
from mantisshrimp.visualize.show_annotation import *


def show_sample(
    sample,
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = None,
    label=True,
    bbox=True,
    mask=True,
    show=False,
    ax: plt.Axes = None,
) -> None:
    show_annotation(
        img=sample["img"] if denormalize_fn is None else denormalize_fn(sample["img"]),
        labels=sample["labels"] if (label and "labels" in sample) else None,
        bboxes=sample["bboxes"] if (bbox and "bboxes" in sample) else None,
        masks=sample["masks"] if (mask and "masks" in sample) else None,
        class_map=class_map,
        ax=ax,
        show=show,
    )


def show_record(
    record: RecordType,
    class_map: Optional[ClassMap] = None,
    label: bool = True,
    bbox: bool = True,
    mask: bool = True,
    ax: plt.Axes = None,
    show: bool = False,
    prepare_record: Optional[callable] = None,
) -> None:
    data_preparer = prepare_record or default_prepare_record
    sample = data_preparer(record)
    show_sample(
        sample=sample,
        label=label,
        bbox=bbox,
        mask=mask,
        class_map=class_map,
        ax=ax,
        show=show,
    )


def show_pred(
    img: np.ndarray,
    pred: dict,
    class_map: Optional[ClassMap] = None,
    denormalize_fn=None,
    label=True,
    bbox=True,
    mask=True,
    show=False,
    ax: plt.Axes = None,
) -> None:
    sample = pred.copy()
    sample["img"] = img

    show_sample(
        sample=sample,
        denormalize_fn=denormalize_fn,
        label=label,
        bbox=bbox,
        mask=mask,
        class_map=class_map,
        show=show,
        ax=ax,
    )


def show_records(
    records: Sequence[RecordType],
    class_map: Optional[ClassMap] = None,
    label: bool = True,
    bbox: bool = True,
    mask: bool = True,
    prepare_record: Optional[callable] = None,
    ncols: int = 1,
    figsize=None,
    show: bool = False,
) -> None:
    partials = [
        partial(
            show_record,
            record=record,
            label=label,
            bbox=bbox,
            mask=mask,
            class_map=class_map,
            prepare_record=prepare_record,
            show=False,
        )
        for record in records
    ]
    plot_grid(partials, ncols=ncols, figsize=figsize, show=show)


def show_samples(
    samples: Sequence[dict],
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = None,
    label=True,
    bbox=True,
    mask=True,
    ncols: int = 1,
    figsize=None,
    show=False,
) -> None:
    partials = [
        partial(
            show_sample,
            sample=sample,
            denormalize_fn=denormalize_fn,
            label=label,
            bbox=bbox,
            mask=mask,
            class_map=class_map,
            show=False,
        )
        for sample in samples
    ]
    plot_grid(partials, ncols=ncols, figsize=figsize, show=show)


def show_preds(
    imgs: Sequence[np.ndarray],
    preds: Sequence[dict],
    class_map: Optional[ClassMap] = None,
    denormalize_fn=None,
    label=True,
    bbox=True,
    mask=True,
    ncols: int = 1,
    figsize=None,
    show=False,
) -> None:
    if not len(imgs) == len(preds):
        raise ValueError(
            f"Number of imgs ({len(imgs)}) should be the same as "
            f"the number of preds ({len(preds)})"
        )

    partials = [
        partial(
            show_pred,
            img=img,
            pred=pred,
            class_map=class_map,
            denormalize_fn=denormalize_fn,
            label=label,
            bbox=bbox,
            mask=mask,
            show=False,
        )
        for img, pred in zip(imgs, preds)
    ]
    plot_grid(partials, ncols=ncols, figsize=figsize, show=show)
