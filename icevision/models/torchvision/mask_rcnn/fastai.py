__all__ = ["learner", "MaskRCNNCallback"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.torchvision.fastai_learner import rcnn_learner
from icevision.models.torchvision.fastai_callbacks import *
from icevision.models.torchvision.mask_rcnn.prediction import *


class MaskRCNNCallback(RCNNCallback):
    def convert_raw_predictions(self, batch, raw_preds):
        return convert_raw_predictions(
            batch=batch,
            raw_preds=raw_preds,
            records=self.learn.records,
            detection_threshold=0.0,
            mask_threshold=0.0,
        )


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
    cbs = [MaskRCNNCallback()] + L(cbs)
    return rcnn_learner(dls=dls, model=model, cbs=cbs, **learner_kwargs)
