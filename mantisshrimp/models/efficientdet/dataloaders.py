__all__ = [
    "build_train_batch",
    "build_valid_batch",
    "train_dataloader",
    "valid_dataloader",
    "build_infer_batch",
]

from mantisshrimp.imports import *
from mantisshrimp.models.utils import transform_collate


def train_dataloader(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    collate_fn = transform_collate(build_batch=build_train_batch, batch_tfms=batch_tfms)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def valid_dataloader(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    collate_fn = transform_collate(build_batch=build_valid_batch, batch_tfms=batch_tfms)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def infer_dataloader(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    collate_fn = partial(build_infer_batch, batch_tfms=batch_tfms)
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


def build_infer_batch(images: List[np.ndarray], batch_tfms=None):
    records = []
    for img in images:
        record = {"img": img}
        record["height"], record["width"], _ = img.shape
        records.append(record)

    if batch_tfms is not None:
        records = batch_tfms(records)

    tensor_imgs, img_sizes = [], []
    for record in records:
        tensor_imgs.append(im2tensor(record["img"]))
        img_sizes.append((record["height"], record["width"]))

    tensor_imgs = torch.stack(tensor_imgs)
    tensor_sizes = tensor(img_sizes, dtype=torch.float)
    tensor_scales = tensor([1] * len(images), dtype=torch.float)

    return tensor_imgs, tensor_scales, tensor_sizes
