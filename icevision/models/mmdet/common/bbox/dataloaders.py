__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
    "train_dl",
    "valid_dl",
    "infer_dl",
]

from icevision.core import *
from icevision.imports import *
from icevision.models.utils import *


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
    records: Sequence[RecordType], batch_tfms=None
) -> Tuple[dict, List[Dict[str, torch.Tensor]]]:
    records = common_build_batch(records=records, batch_tfms=batch_tfms)

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
    records: Sequence[RecordType], batch_tfms=None
) -> Tuple[dict, List[Dict[str, torch.Tensor]]]:
    return build_train_batch(records=records, batch_tfms=batch_tfms)


def build_infer_batch(records, batch_tfms=None):
    records = common_build_batch(records, batch_tfms=batch_tfms)

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
    img = record["img"][:, :, ::-1].copy()
    return im2tensor(img)


def _img_meta(record):
    img_h, img_w, img_c = record["img"].shape

    return {
        # height and width from sample is before padding
        "img_shape": (record["height"], record["width"], img_c),
        "pad_shape": (img_h, img_w, img_c),
        "scale_factor": np.ones(4),  # TODO: is scale factor correct?
    }


def _labels(record):
    if len(record["labels"]) == 0:
        raise RuntimeError("Negative samples still needs to be implemented")
    else:
        # convert background from id 0 to last
        labels = tensor(record["labels"], dtype=torch.int64) - 1
        labels[labels == -1] = record["class_map"].num_classes - 1
        return labels


def _bboxes(record):
    if len(record["labels"]) == 0:
        raise RuntimeError("Negative samples still needs to be implemented")
    else:
        xyxys = [bbox.xyxy for bbox in record["bboxes"]]
        return tensor(xyxys, dtype=torch.float32)
