__all__ = [
    "predict",
    "predict_dl",
    "convert_raw_prediction",
    "convert_raw_predictions",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.models.utils import _predict_dl
from icevision.data import *
from icevision.models.torchvision.faster_rcnn.dataloaders import *


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    model.eval()
    device = device or model_device(model)
    batch = [o.to(device) for o in batch]

    raw_preds = model(*batch)
    return convert_raw_predictions(
        raw_preds=raw_preds, records=records, detection_threshold=detection_threshold
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)
    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
        device=device,
    )


def predict_dl(
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    **predict_kwargs,
):
    return _predict_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        **predict_kwargs,
    )


def convert_raw_predictions(
    raw_preds, records: Sequence[BaseRecord], detection_threshold: float
):
    return [
        convert_raw_prediction(
            raw_pred=raw_pred,
            record=record,
            detection_threshold=detection_threshold,
        )
        for raw_pred, record in zip(raw_preds, records)
    ]


def convert_raw_prediction(
    raw_pred: dict, record: BaseRecord, detection_threshold: float
):
    above_threshold = raw_pred["scores"] >= detection_threshold

    labels = raw_pred["labels"][above_threshold]
    labels = labels.detach().cpu().numpy()

    scores = raw_pred["scores"][above_threshold]
    scores = scores.detach().cpu().numpy()

    boxes = raw_pred["boxes"][above_threshold]
    bboxes = []
    for box_tensor in boxes:
        xyxy = box_tensor.cpu().numpy()
        bbox = BBox.from_xyxy(*xyxy)
        bboxes.append(bbox)

    pred = BaseRecord(
        (
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )
    pred.detection.set_class_map(record.detection.class_map)
    pred.detection.set_scores(scores)
    pred.detection.set_labels_by_id(labels)
    pred.detection.set_bboxes(bboxes)
    pred.detection.above_threshold = above_threshold

    return Prediction(pred=pred, ground_truth=record)
