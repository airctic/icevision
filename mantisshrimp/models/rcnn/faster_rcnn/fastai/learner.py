from mantisshrimp.imports import *
from mantisshrimp.engines.fastai import *
from mantisshrimp.models.rcnn.fastai.learner import rcnn_learner
from mantisshrimp.models.rcnn.faster_rcnn.fastai.callbacks import *


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs
):
    cbs = [FasterRCNNCallback()] + L(cbs)
    return rcnn_learner(dls=dls, model=model, cbs=cbs, **learner_kwargs)
