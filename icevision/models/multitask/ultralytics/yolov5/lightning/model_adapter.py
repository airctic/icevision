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
from icevision.models.multitask.utils.prediction import *
from icevision.models.multitask.ultralytics.yolov5.prediction import (
    convert_raw_predictions,
)
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
        self.post_init()

    def post_init(self):
        pass

    # ======================== TRAINING METHODS ======================== #

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: Tuple[dict, Sequence[RecordType]], batch_idx):
        batch, _ = batch
        if isinstance(batch[0], torch.Tensor):
            (xb, detection_targets, classification_targets) = batch
            step_type = ForwardType.TRAIN

        elif isinstance(batch[0], dict):
            # TODO: Model method not yet implemented
            (detection_data, classification_data) = batch
            detection_targets = detection_data["targets"]
            classification_targets = classification_data["targets"]

            step_type = ForwardType.TRAIN_MULTI_AUG
            raise RuntimeError

        detection_preds, classification_preds = self(xb, step_type=step_type)
        detection_loss = self.compute_loss(detection_preds, detection_targets)[0]

        # Iterate through each head and compute classification losses
        classification_losses = {
            name: head.compute_loss(
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

    def validation_step(self, batch, batch_idx):
        batch, records = batch
        (xb, detection_targets, classification_targets) = batch

        with torch.no_grad():
            # Get bbox preds and unactivated classifier preds, ready to feed to loss funcs
            (inference_det_preds, training_det_preds), classification_preds = self(
                xb, step_type=ForwardType.EVAL
            )

            detection_loss = self.compute_loss(training_det_preds, detection_targets)[0]
            classification_losses = {
                name: head.compute_loss(
                    predictions=classification_preds[name],
                    targets=classification_targets[name],
                )
                for name, head in self.model.classifier_heads.items()
            }
            total_classification_loss = sum(classification_losses.values())
            self.compute_and_log_classification_metrics(
                classification_preds=classification_preds,
                yb=classification_targets,
            )

            # Run activation function on classification predictions
            classification_preds = {
                name: head.postprocess(classification_preds[name])
                for name, head in self.model.classifier_heads.items()
            }

            preds = convert_raw_predictions(
                batch=xb,
                records=records,
                raw_detection_preds=inference_det_preds,
                raw_classification_preds=classification_preds,
                classification_configs=extract_classifier_pred_cfgs(self.model),
                detection_threshold=0.001,
                nms_iou_threshold=0.6,
                keep_images=False,
            )

        self.accumulate_metrics(preds)
        self.log_losses(
            "valid", detection_loss, total_classification_loss, classification_losses
        )

    def validation_epoch_end(self, outs):
        self.finalize_metrics()

    # ======================== LOGGING METHODS ======================== #
    def compute_and_log_classification_metrics(
        self,
        classification_preds: TensorDict,  # activated predictions
        yb: TensorDict,
        on_step: bool = False,
        # prefix: str = "valid",
    ):
        # prefix = f"{prefix}/" if not prefix == "" else ""
        prefix = "valid/"
        for (name, metric), (_, preds) in zip(
            self.classification_metrics.items(), classification_preds.items()
        ):
            self.log(
                f"{prefix}{metric.__class__.__name__.lower()}_{name}",  # accuracy_{task_name}
                metric(preds, yb.type(torch.int)),
                on_step=on_step,
                on_epoch=True,
            )

        for name in self.model.classifier_heads.keys():
            # for name, metric in self.classification_metrics.items():
            metric = getattr(self, f"{name}_accuracy")
            self.log(
                f"{prefix}{metric.__class__.__name__.lower()}__{name}",  # accuracy__shot_framing
                # metric(classification_preds[name], yb_classif[name]),
                metric(classification_preds[name], yb_classif[name].type(torch.int)),
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
            detection_loss=detection_loss,
            classification_total_loss=classification_total_loss,
            **{
                f"classification_loss_{name}": loss
                for name, loss in classification_losses.items()
            },
        )
        for k, v in log_vars.items():
            self.log(f"{mode}/{k}", v.item() if isinstance(v, torch.Tensor) else v)

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def finalize_metrics(self) -> None:
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                self.log(f"{metric.name}/{k}", v)
