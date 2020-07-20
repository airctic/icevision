__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "build_infer_batch",
    "train_dataloader",
    "valid_dataloader",
    "infer_dataloader",
]

from mantisshrimp.imports import *
from mantisshrimp.parsers import *
from mantisshrimp.models.utils import *


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


def infer_dataloader(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dataloader(
        dataset=dataset,
        build_batch=build_infer_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def _build_train_sample(
    record: RecordType,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    image = im2tensor(record["img"])

    target = {
        "labels": tensor(record["labels"], dtype=torch.int64),
        "boxes": tensor([bbox.xyxy for bbox in record["bboxes"]], dtype=torch.float),
    }

    return image, target


def build_train_batch(
    records: List[RecordType], batch_tfms=None
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    records = common_build_batch(records=records, batch_tfms=batch_tfms)

    images, targets = [], []
    for record in records:
        image, target = _build_train_sample(record)
        images.append(image)
        targets.append(target)

    return images, targets


def build_valid_batch(
    records: List[RecordType], batch_tfms=None
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    return build_train_batch(records=records, batch_tfms=batch_tfms)


# TODO: Refactor to add batch_tfms and recieve records
def build_infer_batch(images: List[np.ndarray], batch_tfms=None) -> List[torch.Tensor]:
    return [im2tensor(image) for image in images]
