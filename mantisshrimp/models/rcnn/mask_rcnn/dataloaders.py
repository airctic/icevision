__all__ = [
    "train_dataloader",
    "valid_dataloader",
    "infer_dataloader",
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
]

from mantisshrimp.imports import *
from mantisshrimp.parsers import *
from mantisshrimp.models.utils import transform_dataloader
from mantisshrimp.models.rcnn.faster_rcnn.dataloaders import _build_train_sample
from mantisshrimp.models.rcnn.faster_rcnn.dataloaders import (
    build_infer_batch,
    infer_dataloader,
)


def train_dataloader(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dataloader(
        dataset=dataset,
        build_batch=build_train_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def valid_dataloader(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dataloader(
        dataset=dataset,
        build_batch=build_valid_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def _build_mask_train_sample(record: RecordType):
    image, target = _build_train_sample(record=record)
    target["masks"] = tensor(record["masks"].data, dtype=torch.uint8)

    return image, target


def build_train_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images, targets = [], []
    for record in records:
        image, target = _build_mask_train_sample(record)
        images.append(image)
        targets.append(target)

    return images, targets


def build_valid_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    return build_train_batch(records=records)
