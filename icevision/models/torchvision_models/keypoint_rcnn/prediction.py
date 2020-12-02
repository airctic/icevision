__all__ = ["predict", "predict_dl", "convert_raw_prediction", "convert_raw_predictions"]

from itertools import chain
from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.models.utils import _predict_dl
from icevision.models.torchvision_models.faster_rcnn.prediction import (
    convert_raw_prediction as faster_convert_raw_prediction,
)


@torch.no_grad()
def predict(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    model.eval()
    device = device or model_device(model)
    batch = [o.to(device) for o in batch]

    raw_preds = model(*batch)
    return convert_raw_predictions(
        raw_preds=raw_preds, detection_threshold=detection_threshold
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


def convert_raw_predictions(raw_preds, detection_threshold: float):
    return [
        convert_raw_prediction(
            raw_pred,
            detection_threshold=detection_threshold,
        )
        for raw_pred in raw_preds
    ]


def convert_raw_prediction(raw_pred: dict, detection_threshold: float):
    preds = faster_convert_raw_prediction(
        raw_pred=raw_pred, detection_threshold=detection_threshold
    )

    above_threshold = preds["above_threshold"]
    kps = raw_pred["keypoints"][above_threshold]
    keypoints = []
    for k in kps:
        k = k.cpu().numpy()
        k = list(chain.from_iterable(k))
        # `if sum(k) > 0` prevents empty `KeyPoints` objects to be instantiated.
        # E.g. `k = [0, 0, 0, 0, 0, 0]` is a flattened list of 2 points `(0, 0, 0)` and `(0, 0, 0)`. We don't want a `KeyPoints` object to be created on top of this list.
        if sum(k) > 0:
            keypoints.append(KeyPoints.from_xyv(k, None))

    preds["keypoints"] = keypoints
    preds["keypoints_scores"] = (
        raw_pred["keypoints_scores"][above_threshold].detach().cpu().numpy()
    )

    return preds
