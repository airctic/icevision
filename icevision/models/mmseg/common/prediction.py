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
from icevision.models.interpretation import _move_to_device

import itertools


@torch.no_grad()
def _predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:

    if device and (
        batch["img"][0].device.type != device.type
        or model_device(model).type != device.type
    ):
        _, batch = _move_to_device(None, batch, device.type)
    elif batch["img"][0].device.type != model_device(model).type:
        _, batch = _move_to_device(None, batch, model_device(model).type)

    # Making sure the model is in eval mode
    model.eval()

    raw_preds = model(**batch, return_loss=False)
    preds = convert_raw_predictions(
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

    # Retrieve ground_truth masks from batch, if present (at training time)
    if "gt_semantic_seg" in batch:
        tensor_gts = batch["gt_semantic_seg"].squeeze().chunk(8, dim=0)
    else:
        tensor_gts = [None]

    # Images for inference come in as lists
    if isinstance(batch["img"], list):
        tensor_imgs = [x.chunk(batch["img"][0].shape[0], dim=0) for x in batch["img"]]
        tensor_imgs = list(
            itertools.chain.from_iterable(tensor_imgs)
        )  # Ensuring that all chunks are unrolled into the list
    else:
        # At training time, images are in a single tensor
        tensor_imgs = batch["img"].chunk(batch["img"].shape[0], dim=0)

    if len(tensor_imgs) != len(records):
        raise (Exception("Tensor images and records should have the same length"))

    # TODO: This can be optimized by using looping through first tensor dims behaviour instead of chunking tensors?
    preds = []
    for record, tensor_gt, mask_pred, tensor_image in zip(
        records, cycle(tensor_gts), raw_preds, tensor_imgs
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

        if tensor_gt is not None:

            # This is used at train time to have mask available for metric computation
            if torch.is_tensor(tensor_gt):

                record.segmentation.set_mask_array(
                    MaskArray(tensor_gt.squeeze().cpu().numpy())
                )

        if keep_images:
            record.set_img(tensor_to_image(tensor_image.squeeze()))

            # We load the record to ensure mask data is available
            if len(record.segmentation.masks):

                (h, w, _) = record.img.shape

                record.segmentation.set_mask_array(
                    record.segmentation.masks[0].to_mask(h=h, w=w, pad_dim=False)
                )

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
    device: Optional[torch.device] = None,
    **predict_kwargs,
):

    device = device or model_device(model)

    return _predict_from_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        keep_images=keep_images,
        device=device,
        **predict_kwargs,
    )
