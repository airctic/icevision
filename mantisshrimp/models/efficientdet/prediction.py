__all__ = ["predict", "convert_raw_predictions"]

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
    device = device or model_device(model)
    batch = [o.to(device) for o in batch]

    bench = DetBenchPredict(unwrap_bench(model), config=model.config)
    bench = bench.eval().to(device)

    raw_preds = bench(*batch)
    return convert_raw_predictions(raw_preds, detection_threshold=detection_threshold)


def convert_raw_predictions(raw_preds: torch.Tensor, detection_threshold: float):
    dets = raw_preds.detach().cpu().numpy()
    bs = len(dets)

    batch_scores = dets[..., 4]
    batch_labels = dets[..., 5]
    batch_bboxes = dets[..., 0:4]

    keep = batch_scores > detection_threshold

    batch_scores = batch_scores[keep].reshape(bs, -1)
    batch_labels = batch_labels[keep].reshape(bs, -1)
    batch_bboxes = batch_bboxes[keep].reshape(bs, -1, 4)

    preds = []
    for scores, labels, bboxes in zip(batch_scores, batch_labels, batch_bboxes):
        pred = {
            "scores": scores,
            "labels": labels,
            "bboxes": [BBox.from_xywh(*o) for o in bboxes],
        }
        preds.append(pred)

    return preds
