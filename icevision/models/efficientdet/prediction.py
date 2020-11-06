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
    imgs, img_info = batch
    imgs = imgs.to(device)
    img_info = {k: v.to(device) for k, v in img_info.items()}

    bench = DetBenchPredict(unwrap_bench(model))
    bench = bench.eval().to(device)

    raw_preds = bench(x=imgs, img_info=img_info)
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
            "bboxes": [BBox.from_xyxy(*xyxy) for xyxy in det[:, :4]],
        }
        preds.append(pred)

    return preds
