__all__ = ["predict", "predict_dl", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_dl
from icevision.models.ultralytics.yolov5.dataloaders import *
from yolov5.utils.general import non_max_suppression


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    device = device or model_device(model)

    batch = batch[0].to(device)
    model = model.eval().to(device)

    raw_preds = model(batch)[0]
    preds = convert_raw_predictions(
        raw_preds=raw_preds, records=records, detection_threshold=detection_threshold
    )

    return preds


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
    raw_preds: torch.Tensor, records: Sequence[BaseRecord], detection_threshold: float
) -> List[Prediction]:
    dets = non_max_suppression(
        raw_preds, conf_thres=detection_threshold, iou_thres=0.45
    )
    dets = [d.detach().cpu().numpy() for d in dets]
    preds = []
    for det, record in zip(dets, records):

        pred = BaseRecord(
            (
                ScoresRecordComponent(),
                ImageRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

        pred.detection.set_class_map(record.detection.class_map)
        pred.detection.set_labels_by_id(det[:, 5].astype(int))
        pred.detection.set_bboxes([BBox.from_xyxy(*xyxy) for xyxy in det[:, :4]])
        pred.detection.set_scores(det[:, 4])

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds
