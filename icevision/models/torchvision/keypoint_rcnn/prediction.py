__all__ = [
    "predict",
    "predict_from_dl",
    "convert_raw_prediction",
    "convert_raw_predictions",
]

from itertools import chain
from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl
from icevision.models.torchvision.keypoint_rcnn.dataloaders import *
from icevision.models.torchvision.faster_rcnn.prediction import (
    convert_raw_prediction as faster_convert_raw_prediction,
)


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
):
    model.eval()
    device = device or model_device(model)
    batch = [o.to(device) for o in batch]

    raw_preds = model(*batch)
    return convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
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
    raw_preds,
    records,
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
        for sample, raw_pred, record in zip(zip(*batch), raw_preds, records)
    ]


def convert_raw_prediction(
    sample,
    raw_pred: dict,
    record: BaseRecord,
    detection_threshold: float,
    keep_image: bool = False,
):
    pred = faster_convert_raw_prediction(
        sample=sample,
        raw_pred=raw_pred,
        record=record,
        detection_threshold=detection_threshold,
        keep_image=keep_image,
    )

    above_threshold = pred.detection.above_threshold
    kps = raw_pred["keypoints"][above_threshold]
    keypoints = []
    for k in kps:
        k = k.cpu().numpy()
        k = list(chain.from_iterable(k))
        # `if sum(k) > 0` prevents empty `KeyPoints` objects to be instantiated.
        # E.g. `k = [0, 0, 0, 0, 0, 0]` is a flattened list of 2 points `(0, 0, 0)` and `(0, 0, 0)`. We don't want a `KeyPoints` object to be created on top of this list.
        if sum(k) > 0:
            keypoints.append(KeyPoints.from_xyv(k, None))

    pred.pred.add_component(KeyPointsRecordComponent())
    pred.pred.detection.add_keypoints(keypoints)
    pred.pred.detection.keypoints_scores = (
        raw_pred["keypoints_scores"][above_threshold].detach().cpu().numpy()
    )

    return pred
