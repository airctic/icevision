__all__ = ["MMDetectionCallback"]

from icevision.imports import *
from icevision.engines.fastai import *
from icevision.models.mmdet.utils import *

from icevision.models.mmdet.common.bbox.prediction import convert_raw_predictions


class _ModelWrap(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, xb):
        return self.model.train_step(data=xb, optimizer=None)

    def forward_test(self, xb):
        imgs = xb[0]["img"]
        img_metas = xb[0]["img_metas"]
        return self.model.forward_test(imgs=[imgs], img_metas=[img_metas])


class MMDetectionCallback(fastai.Callback):
    def after_create(self):
        self.learn.model = _ModelWrap(self.model)
        self.model.param_groups = self.model.model.param_groups

    def before_batch(self):
        self.learn.records = self.yb[0]
        self.learn.yb = self.xb

    @abstractmethod
    def convert_raw_predictions(self, batch, raw_preds, records):
        """Convert raw predictions from the model to library standard."""

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
