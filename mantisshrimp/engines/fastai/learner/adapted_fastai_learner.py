__all__ = ["adapted_fastai_learner"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.metrics import *
from mantisshrimp.engines.fastai.imports import *
from mantisshrimp.engines.fastai.adapters import *


# TODO: param_groups fix for efficientdet
def adapted_fastai_learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    metrics=None,
    device=None,
    **learner_kwargs,
):
    # convert dataloaders to fastai
    fastai_dls = []
    for dl in dls:
        if isinstance(dl, DataLoader):
            fastai_dl = convert_dataloader_to_fastai(dl)
        elif isinstance(dl, fastai.DataLoader):
            fastai_dl = dl
        else:
            raise ValueError(f"dl type {type(dl)} not supported")

        fastai_dls.append(fastai_dl)

    device = device or fastai.default_device()
    fastai_dls = fastai.DataLoaders(*fastai_dls).to(device)

    # convert metrics to fastai
    metrics = metrics or []
    fastai_metrics = [
        FastaiMetricAdapter(metric) if isinstance(metric, Metric) else metric
        for metric in metrics
    ]

    return fastai.Learner(
        dls=fastai_dls, model=model, metrics=fastai_metrics, **learner_kwargs,
    )
