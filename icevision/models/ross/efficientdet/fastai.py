__all__ = ["learner"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.ross import efficientdet
from icevision.models.ross.efficientdet.loss_fn import loss_fn


class EfficientDetCallback(fastai.Callback):
    def before_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        self.learn.xb = self.xb[0]
        self.learn.records = self.yb[0]
        self.learn.yb = ()

    def after_pred(self):
        self.learn.yb = [self.learn.xb[1]]
        self.learn.xb = [self.learn.xb[0]]

        if not self.training:
            preds = efficientdet.convert_raw_predictions(
                batch=(*self.xb, *self.yb),
                raw_preds=self.pred["detections"],
                records=self.learn.records,
                detection_threshold=0.0,
            )
            self.learn.converted_preds = preds


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    **learner_kwargs,
):
    """Fastai `Learner` adapted for EfficientDet.

    # Arguments
        dls: `Sequence` of `DataLoaders` passed to the `Learner`.
        The first one will be used for training and the second for validation.
        model: The model to train.
        cbs: Optional `Sequence` of callbacks.
        **learner_kwargs: Keyword arguments that will be internally passed to `Learner`.

    # Returns
        A fastai `Learner`.
    """
    cbs = [EfficientDetCallback()] + L(cbs)

    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=loss_fn,
        **learner_kwargs,
    )

    # HACK: patch AvgLoss (in original, find_bs looks at the first element in dictionary and gives errors)
    class EffDetAvgLoss(fastai.AvgLoss):
        def accumulate(self, learn):
            bs = len(first(learn.yb)["cls"])
            self.total += learn.to_detach(learn.loss.mean()) * bs
            self.count += bs

    recorder = [cb for cb in learn.cbs if isinstance(cb, fastai.Recorder)][0]
    recorder.loss = EffDetAvgLoss()

    return learn
