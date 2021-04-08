__all__ = ["predict", "predict_dl", "convert_raw_prediction", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_dl
from icevision.models.mmdet.common.mask.dataloaders import *

# from icevision.models.mmdet.common.bbox.prediction import (
#     _unpack_raw_bboxes,
# )


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
):
    device = device or model_device(model)
    batch["img"] = [img.to(device) for img in batch["img"]]

    raw_preds = model(return_loss=False, rescale=False, **batch)
    return convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        keep_images=keep_images,
        detection_threshold=detection_threshold,
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)
    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
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
    batch,
    raw_preds,
    records: Sequence[BaseRecord],
    detection_threshold: float,
    keep_images: bool = False,
):
    return [
        convert_raw_prediction(
            sample=sample,
            raw_pred=raw_pred,
            record=record,
            detection_threshold=detection_threshold,
            keep_image=keep_images,
        )
        for sample, raw_pred, record in zip(batch["img"], raw_preds, records)
    ]


def convert_raw_prediction(
    sample,
    raw_pred: dict,
    record: BaseRecord,
    detection_threshold: float,
    keep_image: bool = False,
):
    # convert predictions
    raw_bboxes, raw_masks = raw_pred
    scores, labels, bboxes = _unpack_raw_bboxes(raw_bboxes)

    keep_mask = scores > detection_threshold
    keep_scores = scores[keep_mask]
    keep_labels = labels[keep_mask]
    keep_bboxes = [BBox.from_xyxy(*o) for o in bboxes[keep_mask]]
    set_trace()
    keep_masks = MaskArray(np.vstack(raw_masks)[keep_mask])

    # build prediction
    pred = BaseRecord(
        (
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            MasksRecordComponent(),
        )
    )
    pred.detection.set_class_map(record.detection.class_map)
    pred.detection.set_scores(keep_scores)
    pred.detection.set_labels_by_id(keep_labels)
    pred.detection.set_bboxes(keep_bboxes)
    pred.detection.set_masks(keep_masks)
    pred.above_threshold = keep_mask

    if keep_image:
        image, *_ = sample
        image = image.cpu().numpy().transpose(1, 2, 0)

        pred.set_img(image)
        record.set_img(image)

    return Prediction(pred=pred, ground_truth=record)


def _unpack_raw_bboxes(raw_bboxes):
    stack_raw_bboxes = np.vstack(raw_bboxes)

    scores = stack_raw_bboxes[:, -1]
    bboxes = stack_raw_bboxes[:, :-1]

    # each item in raw_pred is an array of predictions of it's `i` class
    labels = [np.full(o.shape[0], i, dtype=np.int32) for i, o in enumerate(raw_bboxes)]
    labels = np.concatenate(labels)

    return scores, labels, bboxes
