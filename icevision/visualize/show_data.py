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
    display_keypoints: bool = True,
    show: bool = True,
    ax: plt.Axes = None,
    figsize: Tuple[int, int] = None,
    **draw_sample_kwargs,
) -> None:
    img = draw_sample(
        sample=sample,
        class_map=class_map,
        denormalize_fn=denormalize_fn,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
        display_keypoints=display_keypoints,
        **draw_sample_kwargs,
    )
    show_img(img=img, ax=ax, show=show, figsize=figsize)


def show_record(
    record,
    class_map: Optional[ClassMap] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    display_keypoints: bool = True,
    ax: plt.Axes = None,
    show: bool = False,
    figsize: Tuple[int, int] = None,
    **draw_sample_kwargs,
) -> None:
    img = draw_record(
        record=record,
        class_map=class_map,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
        display_keypoints=display_keypoints,
        **draw_sample_kwargs,
    )
    show_img(img=img, ax=ax, show=show, figsize=figsize)


def show_pred(
    pred: Prediction,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    show: bool = True,
    ax: plt.Axes = None,
    figsize=None,
    annotation=None,
    **draw_sample_kwargs,
) -> None:
    actual = draw_sample(
        sample=pred.ground_truth,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
        denormalize_fn=denormalize_fn,
    )

    prediction = draw_pred(
        pred=pred,
        denormalize_fn=denormalize_fn,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
        **draw_sample_kwargs,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].imshow(actual, cmap=None)
    ax[0].set_title("Ground truth")
    ax[1].imshow(prediction, cmap=None)
    ax[1].set_title("Prediction")

    if annotation is None:
        ax[0].set_axis_off()
        ax[1].set_axis_off()
    else:
        ax[0].get_xaxis().set_ticks([])
        ax[0].set_frame_on(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].set_xlabel(annotation[0], ma="left")

        ax[1].get_xaxis().set_ticks([])
        ax[1].set_frame_on(False)
        ax[1].get_yaxis().set_visible(False)
        ax[1].set_xlabel(annotation[1], ma="left")

    if show:
        plt.show()


def show_records(
    records: Sequence[RecordType],
    class_map: Optional[ClassMap] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    ncols: int = 1,
    figsize=None,
    show: bool = False,
    **draw_sample_kwargs,
) -> None:
    partials = [
        partial(
            show_record,
            record=record,
            display_label=display_label,
            display_bbox=display_bbox,
            display_mask=display_mask,
            class_map=class_map,
            show=False,
            **draw_sample_kwargs,
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
    **draw_sample_kwargs,
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
            **draw_sample_kwargs,
        )
        for sample in samples
    ]
    plot_grid(partials, ncols=ncols, figsize=figsize, show=show)


def show_preds(
    preds: Sequence[Prediction],
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    figsize=None,
    show=False,
    annotations=None,
    ncols=None,  # TODO: Not used,
    **draw_sample_kwargs,
) -> None:
    annotations = annotations or [None] * len(preds)
    partials = [
        partial(
            show_pred,
            pred=pred,
            denormalize_fn=denormalize_fn,
            display_label=display_label,
            display_bbox=display_bbox,
            display_mask=display_mask,
            show=False,
            annotation=annotation,
            **draw_sample_kwargs,
        )
        for pred, annotation in zip(preds, annotations)
    ]

    plot_grid(
        partials,
        ncols=2,
        figsize=figsize or (6, 6 * len(preds) / 2 / 0.75),
        show=show,
        axs_per_iter=2,
    )
