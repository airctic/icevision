__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
    "train_dl",
    "valid_dl",
    "infer_dl",
]

from icevision.imports import *
from icevision.core import *
from icevision.models.utils import *


def train_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for training the model.

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
        build_batch=build_train_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def _build_train_sample(
    record: RecordType,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    assert len(record.detection.label_ids) == len(record.detection.bboxes)

    image = im2tensor(record.img)

    # If no labels and bboxes are present, use as negative samples
    if len(record.detection.label_ids) == 0:
        target = torch.zeros((0, 6))
    else:
        labels = tensor(record.detection.label_ids, dtype=torch.int64) - 1

        img_width, img_height = record.width, record.height
        xyxys = [
            bbox.relative_xcycwh(img_width, img_height)
            for bbox in record.detection.bboxes
        ]
        boxes = tensor(xyxys, dtype=torch.float32)

        target = torch.zeros((len(labels), 6))
        target[:, 1:] = torch.cat([labels.unsqueeze(1), boxes], 1)

    return image, target


def build_train_batch(
    records: Sequence[RecordType],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Builds a batch in the format required by the model when training.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be an updated list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_train_batch(records)
    outs = model(*batch)
    ```
    """
    images, targets = [], []
    for i, record in enumerate(records):
        image, target = _build_train_sample(record)
        images.append(image)

        if target.numel() > 0:
            target[:, 0] = i

        targets.append(target)

    return (torch.stack(images, 0), torch.cat(targets, 0)), records


def valid_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    """A `DataLoader` with a custom `collate_fn` that batches items as required for validating the model.

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
        build_batch=build_valid_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def build_valid_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    """Builds a batch in the format required by the model when validating.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be an updated list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_valid_batch(records)
    outs = model(*batch)
    ```
    """
    return build_train_batch(records=records)


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


def build_infer_batch(records: Sequence[RecordType]):
    """Builds a batch in the format required by the model when doing inference.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be an updated list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_infer_batch(records)
    outs = model(*batch)
    ```
    """
    tensor_imgs = [im2tensor(record.img) for record in records]
    tensor_imgs = torch.stack(tensor_imgs)

    return (tensor_imgs,), records
