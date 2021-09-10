__all__ = ["MMDetModelAdapter"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter

from icevision.models.mmdet.common.bbox import convert_raw_predictions


class MMDetModelAdapter(LightningModelAdapter, ABC):
    """Lightning module specialized for EfficientDet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    @abstractmethod
    def convert_raw_predictions(self, batch, raw_preds, records):
        """Convert raw predictions from the model to library standard."""

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        data, samples = batch

        outputs = self.model.train_step(data=data, optimizer=None)

        for k, v in outputs["log_vars"].items():
            self.log(f"train/{k}", v)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        data, records = batch
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.train_step(data=data, optimizer=None)
            raw_preds = self.model.forward_test(
                imgs=[data["img"]], img_metas=[data["img_metas"]]
            )

        preds = self.convert_raw_predictions(
            batch=data, raw_preds=raw_preds, records=records
        )
        self.accumulate_metrics(preds)

        for k, v in outputs["log_vars"].items():
            self.log(f"valid/{k}", v)

        # TODO: is train and eval model automatically set by lighnting?
        self.model.train()

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
