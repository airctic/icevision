__all__ = ["MMSegmentationCallback"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.engines.fastai import *
from icevision.models.mmseg.utils import *
from icevision.models.mmseg.common.prediction import convert_raw_predictions


class _ModelWrap(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, xb):
        return self.model.train_step(data_batch=xb, optimizer=None)

    def forward_test(self, xb):
        imgs = xb[0]["img"]
        img_metas = xb[0]["img_metas"]
        return self.model.forward_test(imgs=[imgs], img_metas=[img_metas])


class MMSegmentationCallback(fastai.Callback):
    def after_create(self):
        self.learn.model = _ModelWrap(self.model)
        self.model.param_groups = self.model.model.param_groups

    def before_batch(self):
        self.learn.records = self.yb[0]
        self.learn.yb = self.xb

    def convert_raw_predictions(self, batch, raw_preds, records):
        return convert_raw_predictions(
            batch=batch, raw_preds=raw_preds, records=records
        )

    # TODO: Understand why this is needed - why can't we get the preds directly from mmseg?
    def after_loss(self):
        if not self.training:
            self.model.eval()
            with torch.no_grad():
                self.learn.raw_preds = self.model.forward_test(self.xb)
            self.model.train()

            # TODO: implement abstract function
            self.learn.converted_preds = self.convert_raw_predictions(
                batch=self.xb[0], raw_preds=self.raw_preds, records=self.records
            )
