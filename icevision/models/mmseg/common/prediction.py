__all__ = [
    "predict",
    "predict_from_dl",
    # "convert_raw_prediction",
    "convert_raw_predictions",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl
from icevision.models.mmseg.common.utils import *
from icevision.models.mmseg.common.dataloaders import *


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    device = model_device(model)

    raw_preds = model(**batch, return_loss=False)
    preds = convert_raw_predictions_no_gt(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        keep_images=keep_images,
    )

    return preds


def convert_raw_predictions(
    batch,
    raw_preds: torch.Tensor,
    records: Sequence[BaseRecord],
    keep_images: bool = False,
) -> List[Prediction]:

    # tensor_images, *_ = batch

    if "gt_semantic_seg" in batch:
        tensor_gts = batch["gt_semantic_seg"].squeeze().chunk(8, dim=0)
    else:
        tensor_gts = []

    tensor_imgs = batch["img"].squeeze().chunk(8, dim=0)

    preds = []
    for record, tensor_gt, mask_pred, tensor_image in zip(
        records, tensor_gts, raw_preds, tensor_imgs
    ):
        pred = BaseRecord(
            (
                ImageRecordComponent(),
                SemanticMaskRecordComponent(),
                ClassMapRecordComponent(task=tasks.segmentation),
            )
        )

        pred.segmentation.set_class_map(record.segmentation.class_map)
        pred.segmentation.set_mask_array(MaskArray(mask_pred))

        if len(tensor_gts):
            record.segmentation.set_mask_array(
                MaskArray(tensor_gt.squeeze().cpu().numpy())
            )

        if keep_images:
            record.set_img(tensor_to_image(tensor_image))

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds


# TODO: Refactor to use a single conversion method?
def convert_raw_predictions_no_gt(
    batch,
    raw_preds: torch.Tensor,
    records: Sequence[BaseRecord],
    keep_images: bool = False,
) -> List[Prediction]:

    tensor_imgs = batch["img"][0].squeeze().chunk(8, dim=0)

    preds = []
    for record, mask_pred, tensor_image in zip(records, raw_preds, tensor_imgs):
        pred = BaseRecord(
            (
                ImageRecordComponent(),
                SemanticMaskRecordComponent(),
                ClassMapRecordComponent(task=tasks.segmentation),
            )
        )

        pred.segmentation.set_class_map(record.segmentation.class_map)
        pred.segmentation.set_mask_array(MaskArray(mask_pred))

        if keep_images:
            record.set_img(tensor_to_image(tensor_image))

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds


def predict(
    model: nn.Module,
    dataset: Dataset,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:

    batch, records = build_infer_batch(dataset)

    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        keep_images=keep_images,
        device=device,
    )


def predict_from_dl(
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    keep_images: bool = False,
    **predict_kwargs,
):
    return _predict_from_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        keep_images=keep_images,
        **predict_kwargs,
    )
