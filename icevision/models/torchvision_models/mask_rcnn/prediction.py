__all__ = ["predict", "predict_dl", "convert_raw_prediction", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.models.utils import _predict_dl
from icevision.models.torchvision_models.faster_rcnn.prediction import (
    convert_raw_prediction as faster_convert_raw_prediction,
)


@torch.no_grad()
def predict(
    model: nn.Module,
    batch: List[torch.Tensor],
    detection_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    model.eval()
    device = device or model_device(model)
    batch = [o.to(device) for o in batch]

    raw_preds = model(*batch)
    return convert_raw_predictions(
        raw_preds=raw_preds,
        detection_threshold=detection_threshold,
        mask_threshold=mask_threshold,
    )


def predict_dl(
    model: nn.Module, infer_dl: DataLoader, show_pbar: bool = True, **predict_kwargs
):
    return _predict_dl(
        predict_fn=predict,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        **predict_kwargs,
    )


def convert_raw_predictions(
    raw_preds: List[dict], detection_threshold: float, mask_threshold: float
):
    return [
        convert_raw_prediction(
            raw_pred=raw_pred,
            detection_threshold=detection_threshold,
            mask_threshold=mask_threshold,
        )
        for raw_pred in raw_preds
    ]


def convert_raw_prediction(
    raw_pred: dict, detection_threshold: float, mask_threshold: float
):
    preds = faster_convert_raw_prediction(
        raw_pred=raw_pred, detection_threshold=detection_threshold
    )

    above_threshold = preds["above_threshold"]
    masks_probs = raw_pred["masks"][above_threshold]
    masks_probs = masks_probs.detach().cpu().numpy()
    # convert probabilities to 0 or 1 based on mask_threshold
    masks = masks_probs > mask_threshold
    masks = MaskArray(masks.squeeze(1))

    return {"masks": masks, **preds}
