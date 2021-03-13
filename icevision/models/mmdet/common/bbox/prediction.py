__all__ = ["predict", "predict_dl", "convert_raw_prediction", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_dl
from icevision.models.mmdet.common.mask.dataloaders import *


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    device = device or model_device(model)
    batch["img"] = [img.to(device) for img in batch["img"]]

    raw_pred = model(return_loss=False, rescale=False, **batch)
    return convert_raw_predictions(
        raw_pred, records=records, detection_threshold=detection_threshold
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
    model: nn.Module, infer_dl: DataLoader, show_pbar: bool = True, **predict_kwargs
):
    return _predict_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        **predict_kwargs,
    )


def convert_raw_predictions(
    raw_preds: Sequence[Sequence[np.ndarray]],
    records: Sequence[BaseRecord],
    detection_threshold: float,
):
    return [
        convert_raw_prediction(
            raw_pred, record=record, detection_threshold=detection_threshold
        )
        for raw_pred, record in zip(raw_preds, records)
    ]


def convert_raw_prediction(
    raw_pred: Sequence[np.ndarray], record: BaseRecord, detection_threshold: float
):
    raw_bboxes = raw_pred
    scores, labels, bboxes = _unpack_raw_bboxes(raw_bboxes)

    keep_mask = scores > detection_threshold
    keep_scores = scores[keep_mask]
    keep_labels = labels[keep_mask]
    keep_bboxes = [BBox.from_xyxy(*o) for o in bboxes[keep_mask]]

    pred = BaseRecord(
        (
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )
    pred.detection.set_class_map(record.detection.class_map)
    pred.detection.set_scores(keep_scores)
    pred.detection.set_labels_by_id(keep_labels)
    pred.detection.set_bboxes(keep_bboxes)
    pred.above_threshold = keep_mask

    return Prediction(pred=pred, ground_truth=record)


def _unpack_raw_bboxes(raw_bboxes):
    stack_raw_bboxes = np.vstack(raw_bboxes)

    scores = stack_raw_bboxes[:, -1]
    bboxes = stack_raw_bboxes[:, :-1]

    # each item in raw_pred is an array of predictions of it's `i` class
    labels = [np.full(o.shape[0], i, dtype=np.int32) for i, o in enumerate(raw_bboxes)]
    labels = np.concatenate(labels)

    return scores, labels, bboxes
