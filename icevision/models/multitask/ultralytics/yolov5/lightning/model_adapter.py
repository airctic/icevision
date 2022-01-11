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
from icevision.models.multitask.ultralytics.yolov5.arch.yolo_hybrid import HybridYOLOV5
from icevision.models.multitask.utils.prediction import *
from icevision.models.multitask.ultralytics.yolov5.prediction import (
    convert_raw_predictions,
)
from icevision.models.multitask.utils.model import ForwardType
from icevision.models.multitask.engines.lightning import MultiTaskLightningModelAdapter
from yolov5.utils.loss import ComputeLoss


class HybridYOLOV5LightningAdapter(MultiTaskLightningModelAdapter):
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
        # batch will ALWAYS return a tuple of 2 elements - batched inputs, records
        tupled_inputs, _ = batch
        if isinstance(tupled_inputs[0], torch.Tensor):
            (xb, detection_targets, classification_targets) = tupled_inputs
            detection_preds, classification_preds = self(
                xb, step_type=ForwardType.TRAIN
            )

        elif isinstance(tupled_inputs[0], dict):
            # TODO: Model method not yet implemented
            data = dict(detection=tupled_inputs[0], classification=tupled_inputs[1])
            detection_targets = data["detection"]["targets"]

            # Go through (a nested dict) each task inside each group and fetch targets
            classification_targets = {}
            for group, datum in data["classification"].items():
                classification_targets.update(datum["targets"])

            detection_preds, classification_preds = self(
                data, step_type=ForwardType.TRAIN_MULTI_AUG
            )

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
        tupled_inputs, records = batch
        (xb, detection_targets, classification_targets) = tupled_inputs

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

            # Run activation function on classification predictions
            classification_preds = {
                name: head.postprocess(classification_preds[name])
                for name, head in self.model.classifier_heads.items()
            }
            self.compute_and_log_classification_metrics(
                classification_preds=classification_preds,
                yb=classification_targets,
            )

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
