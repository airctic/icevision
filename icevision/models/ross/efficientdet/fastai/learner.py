__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.ross.efficientdet.loss_fn import loss_fn
from icevision.models.ross.efficientdet.fastai.callbacks import EfficientDetCallback


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

    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=loss_fn,
        **learner_kwargs,
    )

    return learn
