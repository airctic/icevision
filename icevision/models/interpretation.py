__all__ = [
    "sort_losses",
    "get_stats",
    "get_weighted_sum",
    "add_annotations",
    "get_samples_losses",
    "_move_to_device",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.visualize.show_data import show_preds
from icevision.core.record_components import LossesRecordComponent


def get_weighted_sum(sample, weights):
    loss_weighted = 0
    for loss, weight in weights.items():
        loss_weighted += sample.losses[loss] * weight
    sample.losses["loss_weighted"] = loss_weighted
    return sample


def sort_losses(
    samples: List[dict], preds: List[dict], by: Union[str, dict] = "loss_total"
) -> Tuple[List[dict], List[dict], List[str]]:
    by_copy = deepcopy(by)
    losses_expected = [
        k for k in samples[0].losses.keys() if "loss" in k and k != "loss_total"
    ]
    if "effdet_total_loss" in losses_expected:
        losses_expected.remove("effdet_total_loss")

    if isinstance(by, str):
        loss_check = losses_expected + ["loss_total"]
        assert (
            by in loss_check
        ), f"You must `sort_by` one of the losses. '{by}' is not among {loss_check}"

    if isinstance(by, dict):
        expected = ["weighted"]
        assert (
            by["method"] in expected
        ), f"`method` must be in {expected}, got {by['method']} instead."
        if by["method"] == "weighted":
            losses_passed = set(by["weights"].keys())
            losses_expected = set(losses_expected)
            assert (
                losses_passed == losses_expected
            ), f"You need to pass a weight for each of the losses in {losses_expected}, got {losses_passed} instead."

            samples = [get_weighted_sum(s, by["weights"]) for s in samples]
            by = "loss_weighted"

    l = list(zip(samples, preds))
    l = sorted(l, key=lambda i: i[0].losses[by], reverse=True)
    sorted_samples, sorted_preds = zip(*l)
    annotations = [el.losses["text"] for el in sorted_samples]

    if isinstance(by_copy, dict):
        if by_copy["method"] == "weighted":
            annotations = [
                f"loss_weighted: {round(s.losses['loss_weighted'], 5)}\n" + a
                for a, s in zip(annotations, sorted_samples)
            ]

    return list(sorted_samples), list(sorted_preds), annotations


def get_stats(l: List) -> dict:
    l = np.array(l)
    quants_names = ["1ile", "25ile", "50ile", "75ile", "99ile"]
    quants = np.quantile(l, [0.01, 0.25, 0.5, 0.75, 0.99])
    d = {
        "min": l.min(),
        "max": l.max(),
        "mean": l.mean(),
    }

    q = {k: v for k, v in zip(quants_names, quants)}
    d.update(q)
    return d


def _move_to_device(x, y, device):

    if isinstance(y, list):
        x = [o.to(device) for o in x]
        y = [
            {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in o.items()
            }
            for o in y
        ]
    elif isinstance(
        y, dict
    ):  # this covers the efficientdet case in which `y` is a dict of Union[list, tensor] and not a list of dicts and `x` is a Tensor and not a list of Tensors
        # and the mmdet case in which x and y are merged into a single dict, hence x will be None
        x = x.to(device) if x is not None else x
        for k in y.keys():
            if isinstance(y[k], list):
                y[k] = [
                    o.to(device) if isinstance(o, torch.Tensor) else o for o in y[k]
                ]
            else:
                y[k] = y[k].to(device) if isinstance(y[k], torch.Tensor) else y[k]
    else:
        return x.to(device), y.to(device)
    return x, y


def _prepend_str(d: dict, s: str):
    return {(s + "_" + k if s not in k else k): v for k, v in d.items()}


class Interpretation:
    def __init__(self, losses_dict, valid_dl, infer_dl, predict_from_dl):
        self.losses_dict = losses_dict
        self.valid_dl = valid_dl
        self.infer_dl = infer_dl
        self.predict_from_dl = predict_from_dl

    def _rename_losses(self, losses_dict):
        return losses_dict

    def _sum_losses(self, losses_dict):
        losses_dict["loss_total"] = sum(losses_dict.values())
        return losses_dict

    def _loop(self, dl, model, losses_stats, device):
        samples_plus_losses = []

        with torch.no_grad():
            for (x, y), sample in pbar(dl):
                torch.manual_seed(0)
                x, y = _move_to_device(x, y, device)
                loss = model(x, y)
                loss = {k: float(v.cpu().numpy()) for k, v in loss.items()}
                loss = self._rename_losses(loss)
                loss = self._sum_losses(loss)

                for l in losses_stats.keys():
                    losses_stats[l].append(loss[l])

                loss = _prepend_str(loss, "loss")
                loss_comp = LossesRecordComponent()
                loss_comp.set_losses(loss)
                sample[0].add_component(loss_comp)
                sample[0].set_img(tensor_to_image(x[0]))
                samples_plus_losses.append(sample[0])
        return samples_plus_losses, losses_stats

    def get_losses(
        self,
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
        model.train()
        device = model_device(model)
        losses_stats = self.losses_dict
        dl = self.valid_dl(dataset, batch_size=1, num_workers=0, shuffle=False)

        samples_plus_losses, losses_stats = self._loop(dl, model, losses_stats, device)

        losses_stats = {k: get_stats(v) for k, v in losses_stats.items()}
        losses_stats = _prepend_str(losses_stats, "loss")
        return samples_plus_losses, losses_stats

    def plot_top_losses(
        self,
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
        logger.info(
            f"Losses returned by model: {[l for l in list(_prepend_str(self.losses_dict, 'loss').keys()) if l!='loss_total']}",
        )

        samples, losses_stats = self.get_losses(model, dataset)
        samples = add_annotations(samples)

        dl = self.infer_dl(dataset, batch_size=batch_size)
        preds = self.predict_from_dl(model=model, infer_dl=dl)
        preds = [p.pred for p in preds]

        sorted_samples, sorted_preds, annotations = sort_losses(
            samples, preds, by=sort_by
        )
        assert len(sorted_samples) == len(samples) == len(preds) == len(sorted_preds)

        anns = []
        for ann in annotations:
            ann = ann.split("\n")
            ann1 = "\n".join(ann[:4])
            ann2 = "\n".join(ann[4:])
            anns.append((ann1, ann2))

        sorted_preds = [
            Prediction(pred=p, ground_truth=s)
            for s, p in zip(sorted_samples, sorted_preds)
        ]

        show_preds(
            preds=sorted_preds[:n_samples],
            annotations=anns[:n_samples],
        )
        model.train()
        return sorted_samples, sorted_preds, losses_stats


def add_annotations(samples: List[dict]) -> List[dict]:
    """
    Adds a `text` field to the sample dict to use as annotations when plotting.
    """
    for sample in samples:
        text = ""
        for key in sample.losses.keys():
            if "loss" in key:
                text += f"{key}: {round(sample.losses[key], 5)}\n"
        text += f"IMG: {sample.filepath.name}"
        sample.losses["text"] = text
    return samples


def get_samples_losses(samples_plus_losses):
    def _get_info(sample):
        d = {k: v for k, v in sample.losses.items() if "loss" in k}
        d["filepath"] = sample.filepath
        return d

    return [_get_info(l) for l in samples_plus_losses]
