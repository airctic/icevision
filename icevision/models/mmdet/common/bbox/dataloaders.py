__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
    "train_dl",
    "valid_dl",
    "infer_dl",
]

from icevision.models.mmdet.common.utils import convert_background_from_zero_to_last
from icevision.core import *
from icevision.imports import *
from icevision.models.utils import *
from icevision.utils.imageio import get_number_of_channels
from icevision.utils.imageio import numpy_to_tensor


def train_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dl(
        dataset=dataset,
        build_batch=build_train_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def valid_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dl(
        dataset=dataset,
        build_batch=build_valid_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def infer_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for inferring the model.

    # Arguments
        dataset: Possibly a `Dataset` object, but more generally, any `Sequence` that returns records.
        batch_tfms: Transforms to be applied at the batch level.
        **dataloader_kwargs: Keyword arguments that will be internally passed to a Pytorch `DataLoader`.
        The parameter `collate_fn` is already defined internally and cannot be passed here.

    # Returns
        A Pytorch `DataLoader`.
    """
    return transform_dl(
        dataset=dataset,
        build_batch=build_infer_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def build_train_batch(
    records: Sequence[RecordType],
) -> Tuple[dict, List[Dict[str, torch.Tensor]]]:
    images, labels, bboxes, img_metas = [], [], [], []
    for record in records:
        images.append(_img_tensor(record))
        img_metas.append(_img_meta(record))
        labels.append(_labels(record))
        bboxes.append(_bboxes(record))

    data = {
        "img": torch.stack(images),
        "img_metas": img_metas,
        "gt_labels": labels,
        "gt_bboxes": bboxes,
    }

    return data, records


def build_valid_batch(
    records: Sequence[RecordType],
) -> Tuple[dict, List[Dict[str, torch.Tensor]]]:
    return build_train_batch(records=records)


def build_infer_batch(records):
    imgs, img_metas = [], []
    for record in records:
        imgs.append(_img_tensor(record))
        img_metas.append(_img_meta(record))

    data = {
        "img": [torch.stack(imgs)],
        "img_metas": [img_metas],
    }

    return data, records


def _img_tensor(record):
    # convert from RGB to BGR
    img = record.img[:, :, ::-1].copy()
    return numpy_to_tensor(img)


def _img_meta(record):
    img_c = get_number_of_channels(record.img)

    return {
        "img_shape": (record.img_size.height, record.img_size.width, img_c),
        "pad_shape": (record.img_size.height, record.img_size.width, img_c),
        "scale_factor": np.ones(4),  # TODO: is scale factor correct?
    }


def _labels(record):
    if len(record.detection.label_ids) == 0:
        return torch.empty(0)
    else:
        tensor_label_ids = tensor(record.detection.label_ids)
        labels = convert_background_from_zero_to_last(
            label_ids=tensor_label_ids, class_map=record.detection.class_map
        )
        return labels


def _bboxes(record):
    if len(record.detection.label_ids) == 0:
        return torch.empty((0, 4))
    else:
        xyxys = [bbox.xyxy for bbox in record.detection.bboxes]
        return tensor(xyxys, dtype=torch.float32)
