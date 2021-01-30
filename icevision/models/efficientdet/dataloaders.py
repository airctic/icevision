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
    res = [process_record(record) for record in records]
    d = collate_records(res)
    # passing the size of transformed image to efficientdet, necessary for its own scaling and resizing, see
    # https://github.com/rwightman/efficientdet-pytorch/blob/645d84a6f0cd837703f98f48179a06c354902515/effdet/bench.py#L100
    targets["img_size"] = tensor(
        [_get_image_hw(image) for image in images], dtype=torch.float
    )
    targets["img_scale"] = tensor([1] * len(images), dtype=torch.float)

    return (images, targets), records


def build_infer_batch(records, batch_tfms=None):
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
    records = common_build_batch(records, batch_tfms=batch_tfms)

    res = [process_record(record, get_bbox=False) for record in records]
    images, img_sizes = [], []
    for record in records:
        img = im2tensor(record["img"])
        images.append(img)

        img_sizes.append(_get_image_hw(img))

    images = torch.stack(images)

    tensor_sizes = tensor(img_sizes, dtype=torch.float)
    tensor_scales = tensor([1] * len(records), dtype=torch.float)
    img_info = {"img_size": tensor_sizes, "img_scale": tensor_scales}

    return (images, img_info), records


def _get_image_hw(img: torch.Tensor) -> torch.Tensor:
    return img.shape[-2:]


def process_record(record, get_bbox=True, get_size=True, get_scale=True) -> dict:
    """Extracts information from record and prepares a format required by the EffDet"""
    img_info = dict()
    img_info["img"] = im2tensor(record["img"])
    if get_size:
        img_info["img_size"] = _get_image_hw(img_info["img"])
    if get_scale:
        img_info["img_scale"] = 1.0
    if get_bbox:
        img_info["cls"] = (
            record["labels"] if record["labels"] else [0]
        )  # background and dummy if no label in record
        img_info["bbox"] = (
            [bbox.yxyx for bbox in record["bboxes"]]
            if record["labels"]
            else [[0, 0, 0, 0]]
        )
    return img_info


def collate_records(records) -> dict:
    """Converting list of dictionaries to a dictionary of lists (tensors if possible)"""
    out_dict = {}
    for key, item in records[0].items():
        if isinstance(item, torch.Tensor):
            collate_fn = torch.stack
        elif isinstance(item, torch.Size):
            collate_fn = torch.Tensor
        elif isinstance(item, list):
            collate_fn = lambda L: [torch.Tensor(el) for el in L]
        else:
            collate_fn = torch.Tensor
        out_dict[key] = collate_fn([record[key] for record in records])

    # check if lens are equal
    lens = [len(val) for val in out_dict.values()]
    assert all(x == lens[0] for x in lens), "Collating batch records failed"
    return out_dict
