__all__ = ["ModelAdapter"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter


class ModelAdapter(LightningModelAdapter, ABC):
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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        data, samples = batch

        outputs = self.model.train_step(data=data, optimizer=None)

        for k, v in outputs["log_vars"].items():
            self.log(f"train/{k}", v)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        (xb, yb), samples = batch
        xb = torch.stack(xb)

        # TODO: add number of channels to size mixin?
        bs, img_c, img_h, img_w = xb.shape
        img_metas = [
            {
                # height and width from sample is before padding
                "img_shape": (sample["height"], sample["width"], img_c),
                "pad_shape": (img_h, img_w, img_c),
                "scale_factor": np.ones(4),  # TODO: is scale factor correct?
            }
            for sample in samples
        ]

        data = {
            "img": xb,
            "img_metas": img_metas,
            "gt_bboxes": [o["boxes"] for o in yb],
            "gt_labels": [o["labels"] for o in yb],
        }

        with torch.no_grad():
            raw_preds = self.model.val_step(data=data, optimizer=None)
            set_trace()

            preds = efficientdet.convert_raw_predictions(raw_preds["detections"], 0)
            loss = efficientdet.loss_fn(raw_preds, yb)

        self.accumulate_metrics(samples, preds)

        for k, v in raw_preds.items():
            if "loss" in k:
                self.log(f"valid/{k}", v)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
