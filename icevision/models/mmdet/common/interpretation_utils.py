__all__ = ["sum_losses_mmdet", "loop_mmdet"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.interpretation import _move_to_device


def sum_losses_mmdet(losses_dict):
    loss_ = {}
    for k, v in losses_dict.items():
        if k.startswith("loss"):
            if isinstance(v, torch.Tensor):
                loss_[k] = float(v.cpu().numpy())
            elif isinstance(v, list):
                loss_[k] = sum([float(o.cpu().numpy()) for o in v])

    loss_["loss_total"] = sum(loss_.values())
    return loss_


def loop_mmdet(dl, model, losses_stats, device):
    samples_plus_losses = []

    with torch.no_grad():
        for data, sample in pbar(dl):
            torch.manual_seed(0)
            _, data = _move_to_device(None, data, device)
            loss = model(**data)
            loss = sum_losses_mmdet(loss)

            for l in losses_stats.keys():
                losses_stats[l].append(loss[l])

            sample[0].update(loss)
            samples_plus_losses.append(sample[0])

    return samples_plus_losses, losses_stats
