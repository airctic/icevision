__all__ = ["learner"]

from icevision.imports import *
from icevision.models.mmdet.fastai.learner import mmdetection_learner
from icevision.models.mmdet.common.mask.fastai.callbacks import MaskMMDetectionCallback


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
    cbs = [MaskMMDetectionCallback] + L(cbs)
    return mmdetection_learner(dls=dls, model=model, cbs=cbs, **learner_kwargs)
