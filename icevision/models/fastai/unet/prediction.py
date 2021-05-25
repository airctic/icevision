__all__ = ["predict", "predict_from_dl", "convert_raw_predictions"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl
from icevision.models.fastai.unet.dataloaders import *


@torch.no_grad()
def _predict_batch(
    model: Union[DetBenchTrain, DetBenchPredict],
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    device = model_device(model)

    images, *_ = batch
    images = images.to(device)

    raw_preds = model(images)
    preds = convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
    )

    return preds


def convert_raw_predictions(
    batch,
    raw_preds: torch.Tensor,
    records: Sequence[BaseRecord],
    keep_images: bool = False,
) -> List[Prediction]:
    mask_preds = raw_preds.argmax(dim=1)
    tensor_images, *_ = batch

    preds = []
    for record, tensor_image, mask_pred in zip(records, tensor_images, mask_preds):
        pred = BaseRecord((ImageRecordComponent(), MasksRecordComponent()))
        pred.detection.set_masks(MaskArray(mask_pred.cpu().numpy()))

        if keep_images:
            record.set_img(tensor_to_image(tensor_image))

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds


def predict(
    model: Union[DetBenchTrain, DetBenchPredict],
    dataset: Dataset,
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:

    batch, records = build_infer_batch(dataset)
