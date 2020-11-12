__all__ = ["RCNNCallback", "rcnn_learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.torchvision_models.loss_fn import loss_fn
from icevision.models.torchvision_models.fastai.callbacks import *


def noop_watch(models, criterion=None, log="gradients", log_freq=1000, idx=None):
    return []


def rcnn_learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **kwargs,
):
    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=loss_fn,
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

    is_wandb = [cb for cb in learn.cbs if "WandbCallback" in str(type(cb))]
    if len(is_wandb) == 1:
        logger.warning("Wandb quickfix implemented, for more info check issue #527")
        wandb.watch = noop_watch
    if len(is_wandb) > 1:
        raise ValueError(
            f"It seems you are passing {len(is_wandb)} `WandbCallback` instances to the `learner`. Only 1 is allowed."
        )

    return learn
