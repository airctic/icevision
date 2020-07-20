__all__ = ["learner"]

from mantisshrimp.imports import *
from mantisshrimp.models import efficientdet
from mantisshrimp.models.efficientdet.fastai.callbacks import *
from mantisshrimp.engines.fastai import *


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    param_groups: List[List[nn.Parameter]] = None,
    cbs=None,
    **learner_kwargs,
):
    cbs = [EfficientDetCallback()] + L(cbs)

    # TODO: Figure out splitter
    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=efficientdet.loss_fn,
        param_groups=param_groups,
        **learner_kwargs,
    )

    return learn
