__all__ = ["predict", "predict_dl", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.models.utils import _predict_dl
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
    raw_preds: torch.Tensor, detection_threshold: float
) -> List[dict]:
    dets = raw_preds.detach().cpu().numpy()
    preds = []
    for det in dets:
        if detection_threshold > 0:
            scores = det[:, 4]
            keep = scores > detection_threshold
            det = det[keep]

        pred = {
            "scores": det[:, 4],
            "labels": det[:, 5].astype(int),
            "bboxes": [BBox.from_xywh(*xywh) for xywh in det[:, :4]],
        }
        preds.append(pred)

    return preds
