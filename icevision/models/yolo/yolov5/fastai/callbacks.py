__all__ = ["Yolov5Callback"]

from icevision.engines.fastai import *


class Yolov5Callback(fastai.Callback):
    def before_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        x, y = self.xb[0][0], self.xb[0][1]
        self.learn.xb = [x]
        self.learn.yb = [y]
        self.learn.records = self.yb[0]

    def after_pred(self):
        if not self.training:
            self.learn.pred = self.pred[1]
