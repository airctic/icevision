__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "train_dataloader",
    "valid_dataloader",
]

from mantisshrimp.imports import *
from mantisshrimp.models.utils import transform_collate


def train_dataloader(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    collate_fn = transform_collate(build_batch=build_train_batch, batch_tfms=batch_tfms)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def valid_dataloader(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    collate_fn = transform_collate(build_batch=build_valid_batch, batch_tfms=batch_tfms)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def build_train_batch(records):
    images = []
    targets = {"bbox": [], "cls": []}
    for record in records:
        image = im2tensor(record["img"])
        images.append(image)

        labels = tensor(record["labels"], dtype=torch.float)
        targets["cls"].append(labels)

        bboxes = tensor([bbox.yxyx for bbox in record["bboxes"]], dtype=torch.float)
        targets["bbox"].append(bboxes)
    images = torch.stack(images)

    return images, targets


def build_valid_batch(records):
    images, targets = build_train_batch(records=records)

    batch_size = len(records)
    targets["img_scale"] = tensor([1.0] * batch_size, dtype=torch.float)
    targets["img_size"] = tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float)

    return images, targets
