__all__ = ["show_results", "get_losses", "plot_top_losses"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.base_show_results import base_show_results
from icevision.models.torchvision_models.faster_rcnn.dataloaders import (
    build_infer_batch,
    valid_dl,
)
from icevision.models.torchvision_models.faster_rcnn.prediction import (
    predict,
    get_preds,
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
    device = torch.device("cpu")
    model = model.to(device)
    dl = valid_dl(dataset, batch_size=1, num_workers=0, shuffle=False)

    torch.manual_seed(0)
    samples_plus_losses = []
    losses_stats = {"loss_classifier": [], "loss_box_reg": [], "loss_total": []}

    with torch.no_grad():
        for (x, y), sample in pbar(dl):
            loss = model(x, y)
            loss = {k: v.numpy() for k, v in loss.items()}
            loss["loss_total"] = sum(loss.values())

            text = ""
            for l in losses_stats.keys():
                text += f"{l}: {round(float(loss[l]), 5)}\n"
                losses_stats[l].append(float(loss[l]))
            text += f"IMG: {sample[0]['filepath'].name}"
            loss["text"] = text

            sample[0].update(loss)
            samples_plus_losses.append(sample[0])

    losses_stats = {k: get_stats(v) for k, v in losses_stats.items()}
    return samples_plus_losses, losses_stats


def plot_top_losses(
    model: nn.Module,
    dataset: Dataset,
    sort_by: str = "loss_total",
    n_samples: int = 5,
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

    Returns
    -------
    sorted_samples: those are the samples (dictionaries) contained in the original dataset
                    enriched with losses and sorted by `sort_by`.
    sorted_preds: `model` predictions on `sorted_samples`. Same order as `sorted_samples`.
    losses_stats: dictionary containing losses stats (min, max, mean, 1-25-50-75-99 quantiles)
                  for each one of the losses.
    """

    samples, losses_stats = get_losses(model, dataset)
    _, preds = get_preds(model, dataset)
    sorted_samples, sorted_preds, annotations = sort_losses(samples, preds, by=sort_by)
    show_preds(
        samples=sorted_samples[:n_samples],
        preds=sorted_preds[:n_samples],
        annotations=annotations[:n_samples],
    )

    return sorted_samples, sorted_preds, losses_stats
