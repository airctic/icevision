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


def _build_train_sample(
    record: RecordType,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    assert len(record["labels"]) == len(record["bboxes"])

    image = im2tensor(record["img"])
    target = {}

    # If no labels and bboxes are present, use as negative samples as described in
    # https://github.com/pytorch/vision/releases/tag/v0.6.0
    if len(record["labels"]) == 0:
        target["labels"] = torch.zeros(0, dtype=torch.int64)
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
    else:
        target["labels"] = tensor(record["labels"], dtype=torch.int64)
        xyxys = [bbox.xyxy for bbox in record["bboxes"]]
        target["boxes"] = tensor(xyxys, dtype=torch.float32)

    return image, target


def build_train_batch(
    records: Sequence[RecordType], batch_tfms=None
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Builds a batch in the format required by the model when training.

    # Arguments
        records: A `Sequence` of records.
        batch_tfms: Transforms to be applied at the batch level.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be an updated list
        of the input records with `batch_tfms` applied.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_train_batch(records)
    outs = model(*batch)
    ```
    """
    records = common_build_batch(records=records, batch_tfms=batch_tfms)

    images, targets = [], []
    for record in records:
        image, target = _build_train_sample(record)
        images.append(image)
        targets.append(target)

    return (images, targets), records


def build_valid_batch(
    records: List[RecordType], batch_tfms=None
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    """Builds a batch in the format required by the model when validating.

    # Arguments
        records: A `Sequence` of records.
        batch_tfms: Transforms to be applied at the batch level.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be an updated list
        of the input records with `batch_tfms` applied.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_valid_batch(records)
    outs = model(*batch)
    ```
    """
    return build_train_batch(records=records, batch_tfms=batch_tfms)


def build_infer_batch(dataset: Sequence[RecordType], batch_tfms=None):
    """Builds a batch in the format required by the model when doing inference.

    # Arguments
        records: A `Sequence` of records.
        batch_tfms: Transforms to be applied at the batch level.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be an updated list
        of the input records with `batch_tfms` applied.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_infer_batch(records)
    outs = model(*batch)
    ```
    """
    samples = common_build_batch(dataset, batch_tfms=batch_tfms)

    tensor_imgs = [im2tensor(sample["img"]) for sample in samples]
    tensor_imgs = torch.stack(tensor_imgs)

    return (tensor_imgs,), samples
