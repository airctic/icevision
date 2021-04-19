__all__ = ["Yolov5Callback"]

from icevision.engines.fastai import *
from icevision.models.ultralytics import yolov5


class Yolov5Callback(fastai.Callback):
    def before_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        x, y, records = self.xb[0][0], self.xb[0][1], self.yb
        self.learn.xb = [x]
        self.learn.yb = [y]
        self.learn.records = records[0]

    def after_pred(self):
        if not self.training:
            inference_out, training_out = self.pred[0], self.pred[1]
            self.learn.pred = training_out
            batch = (*self.learn.xb, *self.learn.yb)
            preds = yolov5.convert_raw_predictions(
                batch=batch,
                raw_preds=inference_out,
                records=self.learn.records,
                detection_threshold=0.001,
                nms_iou_threshold=0.6,
            )
            self.learn.converted_preds = preds
