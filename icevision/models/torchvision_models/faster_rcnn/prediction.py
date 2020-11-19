__all__ = ["predict", "predict_dl", "convert_raw_prediction", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.models.utils import _predict_dl
from itertools import chain


@torch.no_grad()
def predict(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    detection_threshold: float = 0.5,
    keypoint_threshold: float = 0.0,
    device: Optional[torch.device] = None,
):
    model.eval()
    device = device or model_device(model)
    batch = [o.to(device) for o in batch]

    raw_preds = model(*batch)
    return convert_raw_predictions(
        raw_preds=raw_preds,
        detection_threshold=detection_threshold,
        keypoint_threshold=keypoint_threshold,
    )


def predict_dl(
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    **predict_kwargs,
):
    return _predict_dl(
        predict_fn=predict,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        **predict_kwargs,
    )


def convert_raw_predictions(
    raw_preds, detection_threshold: float, keypoint_threshold: float = 0.0
):
    return [
        convert_raw_prediction(
            raw_pred,
            detection_threshold=detection_threshold,
            keypoint_threshold=keypoint_threshold,
        )
        for raw_pred in raw_preds
    ]


def convert_raw_prediction(
    raw_pred: dict, detection_threshold: float, keypoint_threshold: float = 0.0
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

    d = {
        "labels": labels,
        "scores": scores,
        "bboxes": bboxes,
        "above_threshold": above_threshold,
    }

    if raw_pred.get("keypoints") is not None:
        # above_threshold_kps = (
        #     (raw_pred["keypoints_scores"] >= keypoint_threshold)
        #     .unsqueeze(2)
        #     .repeat(1, 1, 3)
        # )
        # zeroes = torch.zeros_like(raw_pred["keypoints"])
        # kps = torch.where(above_threshold_kps == True, raw_pred["keypoints"], zeroes)

        # zeroes = torch.zeros_like(raw_pred["keypoints_scores"])
        # kps_scores = torch.where(
        #     (raw_pred["keypoints_scores"] > 0.6) == True,
        #     raw_pred["keypoints_scores"],
        #     zeroes,
        # )
        kps = raw_pred["keypoints"][above_threshold]
        keypoints = []
        for k in kps:
            k = k.cpu().numpy()
            k = list(chain.from_iterable(k))
            if sum(k) > 0:
                keypoints.append(KeyPoints.from_xyv(k, []))

        d["keypoints"] = keypoints
        d["keypoints_scores"] = (
            raw_pred["keypoints_scores"][above_threshold].detach().cpu().numpy()
        )

    return d
