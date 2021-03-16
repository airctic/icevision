__all__ = ["predict", "predict_dl", "convert_raw_prediction", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_dl
from icevision.models.torchvision.mask_rcnn.dataloaders import *
from icevision.models.torchvision.faster_rcnn.prediction import (
    convert_raw_prediction as faster_convert_raw_prediction,
)


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: List[torch.Tensor],
    records: Sequence[BaseRecord],
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
        records=records,
        detection_threshold=detection_threshold,
        mask_threshold=mask_threshold,
    )


def predict(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)
    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
        device=device,
    )


def predict_dl(
    model: nn.Module, infer_dl: DataLoader, show_pbar: bool = True, **predict_kwargs
):
    return _predict_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        **predict_kwargs,
    )


def convert_raw_predictions(
    raw_preds: List[dict],
    records: Sequence[BaseRecord],
    detection_threshold: float,
    mask_threshold: float,
):
    return [
        convert_raw_prediction(
            raw_pred=raw_pred,
            record=record,
            detection_threshold=detection_threshold,
            mask_threshold=mask_threshold,
        )
        for raw_pred, record in zip(raw_preds, records)
    ]


def convert_raw_prediction(
    raw_pred: dict,
    record: Sequence[BaseRecord],
    detection_threshold: float,
    mask_threshold: float,
):
    pred = faster_convert_raw_prediction(
        raw_pred=raw_pred, record=record, detection_threshold=detection_threshold
    )

    above_threshold = pred.detection.above_threshold
    masks_probs = raw_pred["masks"][above_threshold]
    masks_probs = masks_probs.detach().cpu().numpy()
    # convert probabilities to 0 or 1 based on mask_threshold
    masks = masks_probs > mask_threshold
    masks = MaskArray(masks.squeeze(1))

    pred.pred.add_component(MasksRecordComponent())
    pred.detection.set_masks(masks)

    return pred
