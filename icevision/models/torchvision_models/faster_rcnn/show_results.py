__all__ = ["show_results", "get_losses", "plot_top_losses", "add_annotations"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.torchvision_models.faster_rcnn.dataloaders import (
    build_infer_batch,
    valid_dl,
    infer_dl,
)
from icevision.models.torchvision_models.faster_rcnn.prediction import (
    predict,
    predict_dl,
)
from icevision.visualize.show_data import show_preds


def show_results(
    model: nn.Module,
    dataset: Dataset,
    class_map: Optional[ClassMap] = None,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
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
    )


def _move_to_device(x, y, device):
    x = [o.to(device) for o in x]
    y = [
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in o.items()}
        for o in y
    ]
    return x, y


def get_losses(
    model: nn.Module,
    dataset: Dataset,
) -> Tuple[List[dict], dict]:
    """
    Gets a dataset and a model as input and returns losses calculated per image + losses stats.
    In case of faster_rcnn those are loss_classifier, loss_box_reg and loss_total
    (sum of the first 2 plus RPN related losses, e.g. sum of 4 terms).

    Arguments
    ---------
    model: nn.Module
    dataset: Dataset

    Returns
    -------
    samples_plus_losses: those are the samples (dictionaries) contained in the original dataset
                          enriched with losses.
    losses_stats: dictionary containing losses stats (min, max, mean, 1-25-50-75-99 quantiles)
                  for each one of the losses.
    """

    device = model_device(model)
    dl = valid_dl(dataset, batch_size=1, num_workers=0, shuffle=False)

    samples_plus_losses = []
    losses_stats = {
        "loss_classifier": [],
        "loss_box_reg": [],
        "loss_objectness": [],
        "loss_rpn_box_reg": [],
        "loss_total": [],
    }

    with torch.no_grad():
        for (x, y), sample in pbar(dl):
            torch.manual_seed(0)
            x, y = _move_to_device(x, y, device)
            loss = model(x, y)
            loss = {k: float(v.cpu().numpy()) for k, v in loss.items()}
            loss["loss_total"] = sum(loss.values())

            for l in losses_stats.keys():
                losses_stats[l].append(loss[l])

            sample[0].update(loss)
            samples_plus_losses.append(sample[0])

    losses_stats = {k: get_stats(v) for k, v in losses_stats.items()}
    return samples_plus_losses, losses_stats


def add_annotations(samples: List[dict]) -> List[dict]:
    """
    Adds a `text` field to the sample dict to use as annotations when plotting.
    """
    for sample in samples:
        text = ""
        for key in sample.keys():
            if "loss" in key:
                text += f"{key}: {round(sample[key], 5)}\n"
        text += f"IMG: {sample['filepath'].name}"
        sample["text"] = text
    return samples


def plot_top_losses(
    model: nn.Module,
    dataset: Dataset,
    sort_by: str = "loss_total",
    n_samples: int = 5,
    batch_size: int = 8,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Gets a dataset and a model as input. Calculates losses for each sample in the dataset.
    Sorts samples by `sort_by` (e.g. one of the losses). Runs model predictions on samples.
    Plots the top sorted `n_samples` and returns losses + predictions + losses_stats.

    Arguments
    ---------
    model: nn.Module
    dataset: Dataset
    sort_by: (str) the loss to sort samples by
    n_samples: how many samples to show
    batch_size: used when creating the infer dataloader to get model predictions on the dataset

    Returns
    -------
    sorted_samples: those are the samples (dictionaries) contained in the original dataset
                    enriched with losses and sorted by `sort_by`.
    sorted_preds: `model` predictions on `sorted_samples`. Same order as `sorted_samples`.
    losses_stats: dictionary containing losses stats (min, max, mean, 1-25-50-75-99 quantiles)
                  for each one of the losses.
    """
    samples, losses_stats = get_losses(model, dataset)
    samples = add_annotations(samples)

    dl = infer_dl(dataset, batch_size=batch_size)
    _, preds = predict_dl(model=model, infer_dl=dl)

    sorted_samples, sorted_preds, annotations = sort_losses(samples, preds, by=sort_by)
    assert len(sorted_samples) == len(samples) == len(preds) == len(sorted_preds)

    anns = []
    for ann in annotations:
        ann = ann.split("\n")
        ann1 = "\n".join(ann[:3])
        ann2 = "\n".join(ann[3:])
        anns.append((ann1, ann2))

    show_preds(
        samples=sorted_samples[:n_samples],
        preds=sorted_preds[:n_samples],
        annotations=anns[:n_samples],
    )
    model.train()
    return sorted_samples, sorted_preds, losses_stats
