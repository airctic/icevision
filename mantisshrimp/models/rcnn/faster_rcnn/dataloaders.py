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
from mantisshrimp.models.utils import transform_dataloader


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


def build_train_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    images, targets = [], []
    for record in records:
        image = im2tensor(record["img"])
        images.append(image)

        target = {
            "labels": tensor(record["labels"], dtype=torch.int64),
            "boxes": tensor(
                [bbox.xyxy for bbox in record["bboxes"]], dtype=torch.float
            ),
        }
        targets.append(target)

    return images, targets


def build_valid_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    return build_train_batch(records=records)


def build_infer_batch(images: List[np.ndarray]) -> List[torch.Tensor]:
    return [im2tensor(image) for image in images]
