__all__ = ["show_results", "interp"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.ultralytics.yolov5.dataloaders import (
    build_infer_batch,
    valid_dl,
    infer_dl,
)
from icevision.models.ultralytics.yolov5.prediction import (
    predict,
    predict_from_dl,
)
from icevision.models.interpretation import Interpretation

from icevision.models.interpretation import _move_to_device
from icevision.core.record_components import LossesRecordComponent
from yolov5.utils.loss import ComputeLoss


def show_results(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    device: Optional[torch.device] = None,
) -> None:
    return base_show_results(
        predict_fn=predict,
        model=model,
        dataset=dataset,
        num_samples=num_samples,
        ncols=ncols,
        denormalize_fn=denormalize_fn,
        show=show,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
        device=device,
    )


def loop_yolo(dl, model, losses_stats, device):
    samples_plus_losses = []
    compute_loss = ComputeLoss(model)

    with torch.no_grad():
        for (x, y), sample in pbar(dl):
            torch.manual_seed(0)
            x, y = _move_to_device(x, y, device)
            preds = model(x)
            loss = compute_loss(preds, y)[0]
            loss = {
                "loss_yolo": float(loss.cpu().numpy()),
                "loss_total": float(loss.cpu().numpy()),
            }

            for l in losses_stats.keys():
                losses_stats[l].append(loss[l])

            loss_comp = LossesRecordComponent()
            loss_comp.set_losses(loss)
            sample[0].add_component(loss_comp)
            sample[0].set_img(tensor_to_image(x[0]))
            samples_plus_losses.append(sample[0])
    return samples_plus_losses, losses_stats


_LOSSES_DICT = {
    "loss_yolo": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)

interp._loop = loop_yolo
