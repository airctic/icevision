__all__ = ["show_results", "interp"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.mmdet.common.mask.dataloaders import *
from icevision.models.mmdet.common.mask.prediction import *
from icevision.models.interpretation import Interpretation
from icevision.models.mmdet.common.interpretation_utils import (
    sum_losses_mmdet,
    loop_mmdet,
)


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


_LOSSES_DICT = {
    "loss_rpn_cls": [],
    "loss_rpn_bbox": [],
    "loss_cls": [],
    "loss_bbox": [],
    "loss_mask": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_dl=predict_dl,
)

interp._loop = loop_mmdet
