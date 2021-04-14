__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.ultralytics.yolov5.fastai.callbacks import Yolov5Callback
from yolov5.utils.loss import ComputeLoss


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    """Fastai `Learner` adapted for Yolov5.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [Yolov5Callback()] + L(cbs)

    compute_loss = ComputeLoss(model)

    def loss_fn(preds, targets) -> Tensor:
        return compute_loss(preds, targets)[0]

    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=loss_fn,
        **learner_kwargs,
    )

    # HACK: patch AvgLoss (in original, find_bs looks at learn.yb which has shape (N, 6) - with N being number_of_objects_in_image * batch_size. So impossible to retrieve BS)
    class Yolov5AvgLoss(fastai.AvgLoss):
        def accumulate(self, learn):
            bs = len(learn.xb[0])
            self.total += learn.to_detach(learn.loss.mean()) * bs
            self.count += bs

    recorder = [cb for cb in learn.cbs if isinstance(cb, fastai.Recorder)][0]
    recorder.loss = Yolov5AvgLoss()

    return learn
