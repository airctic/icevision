__all__ = ["base_show_results"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.visualize import *
from icevision.data import *


def base_show_results(
    predict_fn: callable,
    build_infer_batch_fn: callable,
    model: nn.Module,
    dataset: Dataset,
    class_map: Optional[ClassMap] = None,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
) -> None:
    samples = random.choices(dataset, k=num_samples)
    batch, samples = build_infer_batch_fn(samples)
    preds = predict_fn(model, batch)

    show_preds(
        samples,
        preds,
        class_map=class_map,
        denormalize_fn=denormalize_fn,
        ncols=ncols,
        show=show,
    )
