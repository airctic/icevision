__all__ = ["learner"]

from icevision.imports import *
from icevision.models.mmseg.fastai.learner import mmsegmentation_learner
from icevision.models.mmseg.common.fastai.callbacks import (
    MaskMMSegmentationCallback,
)


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    """Fastai `Learner` adapted for MMDetection Object Detection models.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [MaskMMSegmentationCallback] + L(cbs)
    return mmsegmentation_learner(dls=dls, model=model, cbs=cbs, **learner_kwargs)
