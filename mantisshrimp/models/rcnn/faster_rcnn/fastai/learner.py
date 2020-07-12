__all__ = ["RCNNCallback", "learner"]

from mantisshrimp.imports import *
from mantisshrimp.engines.fastai import *
from mantisshrimp.models.rcnn.faster_rcnn.loss import loss


class RCNNCallback(fastai.Callback):
    def begin_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        self.learn.xb = (self.xb[0], self.yb[0])
        self.learn.yb = ()

    def after_pred(self):
        self.learn.yb = [self.learn.xb[1]]
        self.learn.xb = [self.learn.xb[0]]

    def begin_validate(self):
        # put model in training mode so we can calculate losses for validation
        self.model.train()

    def after_loss(self):
        if not self.training:
            self.model.eval()
            self.learn.pred = self.model(*self.xb)
            self.model.train()


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **kwargs
):
    cbs = [RCNNCallback()] + L(cbs)

    def model_splitter(model):
        return model.param_groups()

    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=loss,
        splitter=model_splitter,
        **kwargs,
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
