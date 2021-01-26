__all__ = ["show_results", "interp"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.interpretation import Interpretation
from icevision.models.torchvision.mask_rcnn.dataloaders import (
    build_infer_batch,
    valid_dl,
    infer_dl,
)
from icevision.models.torchvision.mask_rcnn.prediction import (
    predict,
    predict_dl,
)


def show_results(
    model: nn.Module,
    dataset: Dataset,
    class_map: Optional[ClassMap] = None,
    detection_threshold: float = 0.5,
    mask_threshold: float = 0.5,
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
        mask_threshold=mask_threshold,
        device=device,
    )


_LOSSES_DICT = {
    "loss_classifier": [],
    "loss_box_reg": [],
    "loss_objectness": [],
    "loss_rpn_box_reg": [],
    "loss_mask": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_dl=predict_dl,
)
