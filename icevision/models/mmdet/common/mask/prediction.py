__all__ = ["predict", "predict_dl", "convert_raw_prediction", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.models.utils import _predict_dl
from icevision.models.mmdet.common.bbox.prediction import (
    _unpack_raw_bboxes,
)


@torch.no_grad()
def predict(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    device = device or model_device(model)
    batch["img"] = [img.cuda() for img in batch["img"]]

    raw_pred = model(return_loss=False, rescale=False, **batch)
    return convert_raw_predictions(raw_pred, detection_threshold=detection_threshold)


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
    raw_preds: Sequence[Sequence[np.ndarray]], detection_threshold: float
):
    return [
        convert_raw_prediction(raw_pred, detection_threshold=detection_threshold)
        for raw_pred in raw_preds
    ]


def convert_raw_prediction(raw_pred: Sequence[np.ndarray], detection_threshold: float):
    raw_bboxes, raw_masks = raw_pred
    scores, labels, bboxes = _unpack_raw_bboxes(raw_bboxes)

    keep_mask = scores > detection_threshold
    keep_scores = scores[keep_mask]
    keep_labels = labels[keep_mask]
    keep_bboxes = [BBox.from_xyxy(*o) for o in bboxes[keep_mask]]
    keep_masks = MaskArray(np.vstack(raw_masks)[keep_mask])

    return {
        "scores": keep_scores,
        "labels": keep_labels,
        "bboxes": keep_bboxes,
        "masks": keep_masks,
    }
