__all__ = ["learner"]

from icevision.imports import *
from icevision.models import efficientdet
from icevision.models.efficientdet.fastai.callbacks import *
from icevision.engines.fastai import *


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    """Fastai `Learner` adapted for EfficientDet.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [EfficientDetCallback()] + L(cbs)

    # TODO: Figure out splitter
    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=efficientdet.loss_fn,
        **learner_kwargs,
    )

    return learn
