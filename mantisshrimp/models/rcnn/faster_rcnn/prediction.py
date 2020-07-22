__all__ = ["predict", "convert_raw_prediction", "convert_raw_predictions"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *


@torch.no_grad()
def predict(
    model: nn.Module,
    batch: List[torch.Tensor],
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    model.eval()
    device = device or model_device(model)
    images = [img.to(device) for img in batch]

    raw_preds = model(images)
    return convert_raw_predictions(
        raw_preds=raw_preds, detection_threshold=detection_threshold
    )


def convert_raw_predictions(raw_preds, detection_threshold: float):
    return [
        convert_raw_prediction(raw_pred, detection_threshold=detection_threshold)
        for raw_pred in raw_preds
    ]


def convert_raw_prediction(raw_pred: dict, detection_threshold: float):
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

    return {
        "labels": labels,
        "scores": scores,
        "bboxes": bboxes,
        "above_threshold": above_threshold,
    }
