__all__ = ["show_results"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.mmdet.common.mask.dataloaders import *
from icevision.models.mmdet.common.mask.prediction import *


def show_results(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.5,
    class_map: Optional[ClassMap] = None,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    device: Optional[torch.device] = None,
) -> None:
    return base_show_results(
        predict_fn=predict,
        build_infer_batch_fn=build_infer_batch,
        model=model,
        dataset=dataset,
        class_map=class_map,
        num_samples=num_samples,
        ncols=ncols,
        denormalize_fn=denormalize_fn,
        show=show,
        detection_threshold=detection_threshold,
        device=device,
    )
