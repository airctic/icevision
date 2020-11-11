__all__ = ["RCNNCallback"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.torchvision_models import faster_rcnn


class RCNNCallback(fastai.Callback, ABC):
    @abstractmethod
    def convert_raw_predictions(self, raw_preds):
        """Convert raw predictions from the model to library standard."""

    def before_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        self.learn.xb = self.xb[0]
        self.learn.records = self.yb[0]
        self.learn.yb = ()

    def after_pred(self):
        self.learn.yb = [self.learn.xb[1]]
        self.learn.xb = [self.learn.xb[0]]

    def before_validate(self):
        # put model in training mode so we can calculate losses for validation
        self.model.train()

    def after_loss(self):
        if not self.training:
            self.model.eval()
            self.learn.pred = self.model(*self.xb)
            self.model.train()

            preds = self.convert_raw_predictions(raw_preds=self.pred)
            self.learn.converted_preds = preds
