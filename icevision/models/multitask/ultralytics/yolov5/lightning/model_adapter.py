# Modified from `icevision.models.mmdet.lightning.model_adapter`
# NOTE `torchmetrics` comes installed with `pytorch-lightning`
# We could in theory also do `pl.metrics`

from icevision.models.multitask.classification_heads.head import TensorDict
import torchmetrics as tm
import pytorch_lightning as pl

from icevision.imports import *
from icevision.metrics import *
from icevision.core import *

from loguru import logger
from icevision.models.multitask.ultralytics.yolov5.yolo_hybrid import HybridYOLOV5
from icevision.models.multitask.utils.model import ForwardType
from yolov5.utils.loss import ComputeLoss


class HybridYOLOV5LightningAdapter(pl.LightningModule, ABC):
    """ """

    def __init__(
        self,
        model: HybridYOLOV5,
        metrics: List[Metric] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.metrics = metrics or []
        self.model = model
        self.debug = debug
        self.compute_loss = ComputeLoss(model)

        self.classification_metrics = nn.ModuleDict()
        for name, head in model.classifier_heads.items():
            if head.multilabel:
                thresh = head.thresh if head.thresh is not None else 0.5
                metric = tm.Accuracy(threshold=thresh, subset_accuracy=True)
            else:
                metric = tm.Accuracy(threshold=0.01, top_k=1)
            self.classification_metrics[name] = metric
            setattr(self, f"{name}_accuracy", metric)
        self.post_init()

    def post_init(self):
        pass

    def training_step(self, batch: Tuple[dict, Sequence[RecordType]], batch_idx):
        batch, _ = batch
        if isinstance(batch[0], torch.Tensor):
            (xb, detection_targets, classification_targets) = batch
            step_type = ForwardType.TRAIN

        elif isinstance(batch[0], dict):
            (detection_data, classification_data) = batch
            detection_targets = detection_data["targets"]

            step_type = ForwardType.TRAIN_MULTI_AUG
            raise RuntimeError

        detection_preds, classification_preds = self(xb, step_type=step_type)
        detection_loss = self.compute_loss(detection_preds, detection_targets)[0]

        # Iterate through each head and compute classification losses
        classification_losses = {
            head.compute_loss(
                predictions=classification_preds[name],
                targets=classification_targets[name],
            )
            for name, head in self.model.classifier_heads.items()
        }
        total_classification_loss = sum(classification_losses.values())

        self.log_losses(
            "train", detection_loss, total_classification_loss, classification_losses
        )

        return detection_loss + total_classification_loss

    def log_losses(
        self,
        mode: str,
        detection_loss: Tensor,
        classification_total_loss: Tensor,
        classification_losses: TensorDict,
    ):
        log_vars = dict(
            detection_loss=detection_loss,
            classification_total_loss=classification_total_loss,
            **{
                f"classification_{name}": loss
                for name, loss in classification_losses.items()
            },
        )
        for k, v in log_vars.items():
            self.log(f"{mode}/{k}", v.item() if isinstance(v, torch.Tensor) else v)

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    # ======================== TRAINING METHODS ======================== #

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ======================== LOGGING METHODS ======================== #

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def finalize_metrics(self) -> None:
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                self.log(f"{metric.name}/{k}", v)
