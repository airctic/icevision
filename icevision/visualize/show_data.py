__all__ = [
    "show_sample",
    "show_record",
    "show_pred",
    "show_samples",
    "show_records",
    "show_preds",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.parsers import *
from icevision.data import *
from icevision.visualize.draw_data import *


def show_sample(
    sample,
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    show: bool = True,
    ax: plt.Axes = None,
) -> None:
    img = draw_sample(
        sample=sample,
        class_map=class_map,
        denormalize_fn=denormalize_fn,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
    )
    show_img(img=img, ax=ax, show=show)


def show_record(
    record,
    class_map: Optional[ClassMap] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    ax: plt.Axes = None,
    show: bool = False,
    prepare_record: Optional[callable] = None,
) -> None:
    img = draw_record(
        record=record,
        class_map=class_map,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
        prepare_record=prepare_record,
    )
    show_img(img=img, ax=ax, show=show)


def show_pred(
    img: np.ndarray,
    pred: dict,
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    show: bool = True,
    ax: plt.Axes = None,
) -> None:
    img = draw_pred(
        img=img,
        pred=pred,
        class_map=class_map,
        denormalize_fn=denormalize_fn,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
    )
    show_img(img=img, ax=ax, show=show)


def show_records(
    records: Sequence[RecordType],
    class_map: Optional[ClassMap] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    prepare_record: Optional[callable] = None,
    ncols: int = 1,
    figsize=None,
    show: bool = False,
) -> None:
    partials = [
        partial(
            show_record,
            record=record,
            display_label=display_label,
            display_bbox=display_bbox,
            display_mask=display_mask,
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
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    ncols: int = 1,
    figsize=None,
    show=False,
) -> None:
    partials = [
        partial(
            show_sample,
            sample=sample,
            denormalize_fn=denormalize_fn,
            display_label=display_label,
            display_bbox=display_bbox,
            display_mask=display_mask,
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
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
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
            display_label=display_label,
            display_bbox=display_bbox,
            display_mask=display_mask,
            show=False,
        )
        for img, pred in zip(imgs, preds)
    ]
    plot_grid(partials, ncols=ncols, figsize=figsize, show=show)
