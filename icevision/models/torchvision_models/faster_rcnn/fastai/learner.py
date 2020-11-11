from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.torchvision_models.fastai.learner import rcnn_learner
from icevision.models.torchvision_models.faster_rcnn.fastai.callbacks import *


def learner(
    dls: Sequence[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs: Optional[Sequence[fastai.Callback]] = None,
    **learner_kwargs
) -> fastai.Learner:
    """Fastai `Learner` adapted for Faster RCNN.

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
