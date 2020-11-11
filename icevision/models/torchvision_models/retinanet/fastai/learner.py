__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.torchvision_models.fastai.learner import rcnn_learner
from icevision.models.torchvision_models.retinanet.fastai.callbacks import *


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs
):
    """Fastai `Learner` adapted for Mask RCNN.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [RetinanetCallback()] + L(cbs)
    return rcnn_learner(dls=dls, model=model, cbs=cbs, **learner_kwargs)
