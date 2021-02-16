__all__ = ["show_results", "interp"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.mmdet.common.bbox.dataloaders import *
from icevision.models.mmdet.common.bbox.prediction import *
from icevision.models.interpretation import Interpretation, _move_to_device


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
    "loss_total": [],
}


def _sum_losses(losses_dict):
    loss_ = {}
    for k, v in losses_dict.items():
        if k.startswith("loss"):
            if isinstance(v, torch.Tensor):
                loss_[k] = float(v.cpu().numpy())
            elif isinstance(v, list):
                loss_[k] = sum([float(o.cpu().numpy()) for o in v])

    loss_["loss_total"] = sum(loss_.values())
    return loss_


def _loop_mmdet_bbox(dl, model, losses_stats, device):
    samples_plus_losses = []

    with torch.no_grad():
        for data, sample in pbar(dl):
            torch.manual_seed(0)
            _, data = _move_to_device(None, data, device)
            loss = model(**data)
            loss = _sum_losses(loss)

            for l in losses_stats.keys():
                losses_stats[l].append(loss[l])

            sample[0].update(loss)
            samples_plus_losses.append(sample[0])

    return samples_plus_losses, losses_stats


interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_dl=predict_dl,
)

interp._loop = _loop_mmdet_bbox
