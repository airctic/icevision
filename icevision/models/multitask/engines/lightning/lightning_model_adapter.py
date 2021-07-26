__all__ = ["MultiTaskLightningModelAdapter"]

import pytorch_lightning as pl
from icevision.imports import *
from icevision.metrics import *
from icevision.engines.lightning import LightningModelAdapter
from icevision.models.multitask.utils.dtypes import *


class MultiTaskLightningModelAdapter(LightningModelAdapter):
    def compute_and_log_classification_metrics(
        self,
        classification_preds: TensorDict,  # activated predictions
        yb: TensorDict,
        on_step: bool = False,
        # prefix: str = "valid",
    ):
        if not set(classification_preds.keys()) == set(yb.keys()):
            raise RuntimeError(
                f"Mismatch between prediction and target items. Predictions have "
                f"{classification_preds.keys()} keys and targets have {yb.keys()} keys"
            )
        # prefix = f"{prefix}/" if not prefix == "" else ""
        prefix = "valid/"
        for (name, metric), (_, preds) in zip(
            self.classification_metrics.items(), classification_preds.items()
        ):
            self.log(
                f"{prefix}{metric.__class__.__name__.lower()}_{name}",  # accuracy_{task_name}
                metric(preds, yb[name].type(torch.int)),
                on_step=on_step,
                on_epoch=True,
            )

    def log_losses(
        self,
        mode: str,
        detection_loss: Tensor,
        classification_total_loss: Tensor,
        classification_losses: TensorDict,
    ):
        log_vars = dict(
            total_loss=detection_loss + classification_total_loss,
            detection_loss=detection_loss,
            classification_total_loss=classification_total_loss,
            **{
                f"classification_loss_{name}": loss
                for name, loss in classification_losses.items()
            },
        )
        for k, v in log_vars.items():
            self.log(f"{mode}/{k}", v.item() if isinstance(v, torch.Tensor) else v)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
