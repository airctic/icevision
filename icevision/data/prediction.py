__all__ = ["Prediction"]

from icevision.imports import *
from icevision.core import *


class Prediction:
    def __init__(self, pred: BaseRecord, ground_truth: Optional[BaseRecord] = None):
        self.ground_truth = ground_truth
        self.pred = pred

        # TODO: imageid, img_size and stuff has to be set even if ground_truth is not stored
        if ground_truth is not None:
            pred.set_imageid(ground_truth.imageid)
            pred.set_img_size(ground_truth.original_img_size, original=True)
            pred.set_img_size(ground_truth.img_size)
            pred.set_img(ground_truth.img)

    def __getattr__(self, name):
        if name == "pred":
            raise AttributeError

        return getattr(self.pred, name)
