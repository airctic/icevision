__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
    "train_dl",
    "valid_dl",
    "infer_dl",
]

from icevision.imports import *
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


def build_train_batch(records):
    """Builds a batch in the format required by the model when training.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be a list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_train_batch(records)
    outs = model(*batch)
    ```
    """
    batch_images, batch_bboxes, batch_classes = zip(
        *(process_train_record(record) for record in records)
    )

    # convert to tensors
    batch_images = torch.stack(batch_images)
    batch_bboxes = [tensor(bboxes, dtype=torch.float32) for bboxes in batch_bboxes]
    batch_classes = [tensor(classes, dtype=torch.float32) for classes in batch_classes]

    # convert to EffDet interface
    targets = dict(bbox=batch_bboxes, cls=batch_classes)

    return (batch_images, targets), records


def build_valid_batch(records):
    """Builds a batch in the format required by the model when validating.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be a list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_valid_batch(records)
    outs = model(*batch)
    ```
    """
    (batch_images, targets), records = build_train_batch(records)

    # convert to EffDet interface, when not training, dummy size and scale is required
    targets = dict(img_size=None, img_scale=None, **targets)

    return (batch_images, targets), records


def build_infer_batch(records):
    """Builds a batch in the format required by the model when doing inference.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be a list
        of the input records.
    Use the result of this function to feed the model.
    ```python
    batch, records = build_infer_batch(records)
    outs = model(*batch)
    ```
    """
    batch_images, batch_sizes, batch_scales = zip(
        *(process_infer_record(record) for record in records)
    )

    # convert to tensors
    batch_images = torch.stack(batch_images)
    batch_sizes = tensor(batch_sizes, dtype=torch.float32)
    batch_scales = tensor(batch_scales, dtype=torch.float32)

    # convert to EffDet interface
    targets = dict(img_size=batch_sizes, img_scale=batch_scales)

    return (batch_images, targets), records


def process_train_record(record) -> tuple:
    """Extracts information from record and prepares a format required by the EffDet training"""
    image = im2tensor(record.img)
    # background and dummy if no label in record
    classes = record.detection.label_ids if record.detection.label_ids else [0]
    bboxes = (
        [bbox.yxyx for bbox in record.detection.bboxes]
        if len(record.detection.label_ids) > 0
        else [[0, 0, 0, 0]]
    )
    return image, bboxes, classes


def process_infer_record(record) -> tuple:
    """Extracts information from record and prepares a format required by the EffDet inference"""
    image = im2tensor(record.img)
    image_size = image.shape[-2:]
    image_scale = 1.0

    return image, image_size, image_scale
