from mantisshrimp.imports import *
from mantisshrimp.engines.fastai import *
from mantisshrimp.models.rcnn.fastai.learner import rcnn_learner
from mantisshrimp.models.rcnn.faster_rcnn.fastai.callbacks import *


def learner(
    dls: Sequence[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs: Optional[Sequence[fastai.Callback]] = None,
    **learner_kwargs
) -> fastai.Learner:
    """ Fastai `Learner` adapted for Faster RCNN.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [FasterRCNNCallback()] + L(cbs)
    return rcnn_learner(dls=dls, model=model, cbs=cbs, **learner_kwargs)
