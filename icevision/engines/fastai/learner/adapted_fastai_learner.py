__all__ = ["adapted_fastai_learner"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.fastai.imports import *
from icevision.engines.fastai.adapters import *


# TODO: param_groups fix for efficientdet
def adapted_fastai_learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    metrics=None,
    device=None,
    splitter=None,
    **learner_kwargs,
) -> fastai.Learner:
    # convert dataloaders to fastai
    fastai_dls = convert_dataloaders_to_fastai(dls=dls, device=device)

    # convert metrics to fastai
    metrics = metrics or []
    fastai_metrics = [
        FastaiMetricAdapter(metric) if isinstance(metric, Metric) else metric
        for metric in metrics
    ]

    if splitter == None:
        if hasattr(model, "param_groups"):

            def splitter(model):
                return model.param_groups()

        else:
            raise ValueError(
                "If the parameter `splitter` is not specified, "
                "the model should define a method called `param_groups`"
            )

    learn = fastai.Learner(
        dls=fastai_dls,
        model=model,
        metrics=fastai_metrics,
        splitter=splitter,
        **learner_kwargs,
    )
    learn.freeze()
    return learn
