__all__ = ["predict", "predict_from_dl", "convert_raw_predictions", "end2end_detect"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl
from icevision.models.ultralytics.yolov5.dataloaders import *
from yolov5.utils.general import non_max_suppression
from icevision.models.inference import *


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    # device issue addressed on discord: https://discord.com/channels/735877944085446747/770279401791160400/832361687855923250
    if device is not None:
        raise ValueError(
            "For YOLOv5 device can only be specified during model creation, "
            "for more info take a look at the discussion here: "
            "https://discord.com/channels/735877944085446747/770279401791160400/832361687855923250"
        )
    grid = model.model[-1].grid[-1]
    # if `grid.numel() == 1` it means the grid isn't initialized yet and we can't
    # trust it's device (will always be CPU)
    device = grid.device if grid.numel() > 1 else model_device(model)

    batch = batch[0].to(device)
    model = model.eval().to(device)

    raw_preds = model(batch)[0]
    return convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
        keep_images=keep_images,
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)
    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
        keep_images=keep_images,
        device=device,
    )


def predict_from_dl(
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    keep_images: bool = False,
    **predict_kwargs,
):
    return _predict_from_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        keep_images=keep_images,
        **predict_kwargs,
    )


def convert_raw_predictions(
    batch,
    raw_preds: torch.Tensor,
    records: Sequence[BaseRecord],
    detection_threshold: float,
    nms_iou_threshold: float,
    keep_images: bool = False,
) -> List[Prediction]:
    dets = non_max_suppression(
        raw_preds, conf_thres=detection_threshold, iou_thres=nms_iou_threshold
    )
    dets = [d.detach().cpu().numpy() for d in dets]
    preds = []
    for det, record, tensor_image in zip(dets, records, batch):

        pred = BaseRecord(
            (
                ScoresRecordComponent(),
                ImageRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

        pred.detection.set_class_map(record.detection.class_map)
        pred.detection.set_labels_by_id(det[:, 5].astype(int) + 1)
        pred.detection.set_bboxes([BBox.from_xyxy(*xyxy) for xyxy in det[:, :4]])
        pred.detection.set_scores(det[:, 4])

        if keep_images:
            record.set_img(tensor_to_image(tensor_image))

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds


end2end_detect = partial(_end2end_detect, predict_fn=predict)
