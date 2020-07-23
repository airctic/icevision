__all__ = ["RCNNCallback", "rcnn_learner"]

from mantisshrimp.imports import *
from mantisshrimp.engines.fastai import *
from mantisshrimp.models.rcnn.loss_fn import loss_fn
from mantisshrimp.models.rcnn.fastai.callbacks import *


def rcnn_learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **kwargs
):
    learn = adapted_fastai_learner(
        dls=dls, model=model, cbs=cbs, loss_func=loss_fn, **kwargs,
    )

    # HACK: patch AvgLoss (in original, find_bs gives errors)
    class RCNNAvgLoss(fastai.AvgLoss):
        def accumulate(self, learn):
            bs = len(learn.yb)
            self.total += fastai.to_detach(learn.loss.mean()) * bs
            self.count += bs

    recorder = [cb for cb in learn.cbs if isinstance(cb, fastai.Recorder)][0]
    recorder.loss = RCNNAvgLoss()

    return learn
