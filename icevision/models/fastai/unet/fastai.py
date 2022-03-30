__all__ = ["learner", "UnetCallback"]


from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.fastai.unet.prediction import convert_raw_predictions


class UnetCallback(fastai.Callback):
    def before_batch(self):
        assert len(self.xb) == len(self.yb) == 1

        self.learn.records = self.yb[0]
        self.learn.yb = (self.xb[0][1],)
        self.learn.xb = (self.xb[0][0],)

    def after_pred(self):
        if not self.training:

            with torch.no_grad():

                preds = convert_raw_predictions(
                    batch=(*self.learn.xb, *self.learn.yb),
                    raw_preds=self.pred,
                    records=self.learn.records,
                )
                self.learn.converted_preds = preds


def learner(
    dls: List[Union[DataLoader, fastai.DataLoader]],
    model: nn.Module,
    cbs=None,
    loss_func=fastai.CrossEntropyLossFlat(axis=1),
    **kwargs,
):
    cbs = L(UnetCallback()) + L(cbs)

    learn = adapted_fastai_learner(
        dls=dls,
        model=model,
        cbs=cbs,
        loss_func=loss_func,
        **kwargs,
    )

    return learn
