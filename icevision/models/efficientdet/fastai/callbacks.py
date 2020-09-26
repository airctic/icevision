__all__ = ["EfficientDetCallback"]

from icevision.models import efficientdet
from icevision.engines.fastai import *


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
            preds = efficientdet.convert_raw_predictions(self.pred["detections"], 0)
            self.learn.converted_preds = preds
