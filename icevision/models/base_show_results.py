__all__ = ["base_show_results"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.visualize import *
from icevision.data import *


def base_show_results(
    predict_fn: callable,
    model: nn.Module,
    dataset: Dataset,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    **predict_kwargs,
) -> None:
    records = random.choices(dataset, k=num_samples)
    preds = predict_fn(model, records, **predict_kwargs)

    show_preds(
        preds,
        denormalize_fn=denormalize_fn,
        ncols=ncols,
        show=show,
    )
