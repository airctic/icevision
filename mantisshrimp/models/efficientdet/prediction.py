__all__ = ["predict", "convert_raw_prediction"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *
from effdet import DetBenchTrain, DetBenchPredict, unwrap_bench


@torch.no_grad()
def predict(
    model: Union[DetBenchTrain, DetBenchPredict],
    batch: Sequence[torch.Tensor],
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    bench = DetBenchPredict(unwrap_bench(model), config=model.config).eval()
    device = device or model_device(bench)
    batch = [o.to(device) for o in batch]

    raw_preds = bench(*batch)
    return [
        convert_raw_prediction(raw_pred, detection_threshold=detection_threshold)
        for raw_pred in raw_preds
    ]


def convert_raw_prediction(
    raw_pred: torch.Tensor, detection_threshold: float
) -> Dict[str, any]:
    pred = {"scores": [], "labels": [], "bboxes": []}
    for det in raw_pred:
        score = det[4].item()
        if score > detection_threshold:
            label = det[5].int().item()
            bbox = BBox.from_xywh(*det[0:4].tolist())

            pred["scores"].append(score)
            pred["labels"].append(label)
            pred["bboxes"].append(bbox)

    return pred
