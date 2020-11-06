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


def build_train_batch(records, batch_tfms=None):
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
    records = common_build_batch(records, batch_tfms=batch_tfms)

    images = []
    targets = {"bbox": [], "cls": []}
    for record in records:
        image = im2tensor(record["img"])
        images.append(image)

        if len(record["labels"]) == 0:
            targets["cls"].append(tensor([0], dtype=torch.float))
            targets["bbox"].append(tensor([[0, 0, 0, 0]], dtype=torch.float))
        else:
            labels = tensor(record["labels"], dtype=torch.float)
            targets["cls"].append(labels)

            bboxes = tensor([bbox.yxyx for bbox in record["bboxes"]], dtype=torch.float)
            targets["bbox"].append(bboxes)

    images = torch.stack(images)

    return (images, targets), records


def build_valid_batch(records, batch_tfms=None):
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
    (images, targets), records = build_train_batch(
        records=records, batch_tfms=batch_tfms
    )

    img_sizes = [(r["height"], r["width"]) for r in records]
    targets["img_size"] = tensor(img_sizes, dtype=torch.float)

    targets["img_scale"] = tensor([1] * len(records), dtype=torch.float)

    return (images, targets), records


def build_infer_batch(dataset, batch_tfms=None):
    """Builds a batch in the format required by the model when doing inference.

    # Arguments
        records: A `Sequence` of records.
        batch_tfms: Transforms to be applied at the batch level.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, targets)`,
        in the input format required by the model. The second will be an updated list
        of the input records with `batch_tfms` applied. # Examples
    Use the result of this function to feed the model.
    ```python
    batch, records = build_infer_batch(records)
    outs = model(*batch)
    ```
    """
    samples = common_build_batch(dataset, batch_tfms=batch_tfms)

    tensor_imgs, img_sizes = [], []
    for record in samples:
        tensor_imgs.append(im2tensor(record["img"]))
        img_sizes.append((record["height"], record["width"]))

    tensor_imgs = torch.stack(tensor_imgs)
    tensor_sizes = tensor(img_sizes, dtype=torch.float)
    tensor_scales = tensor([1] * len(samples), dtype=torch.float)
    img_info = {"img_size": tensor_sizes, "img_scale": tensor_scales}

    return (tensor_imgs, img_info), samples
