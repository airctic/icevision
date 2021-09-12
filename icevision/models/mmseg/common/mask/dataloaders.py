__all__ = ["train_dl", "valid_dl", "infer_dl", "build_train_batch", "build_infer_batch"]


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
    return train_dl(dataset, batch_tfms=None, **dataloader_kwargs)


def infer_dl(dataset, batch_tfms=None, **dataloader_kwargs) -> DataLoader:
    return transform_dl(
        dataset=dataset,
        build_batch=build_infer_batch,
        batch_tfms=batch_tfms,
        **dataloader_kwargs
    )


def _img_meta_mask(record):
    img_meta = _img_meta(record)
    img_meta["ori_shape"] = img_meta["pad_shape"]
    return img_meta


def _img_meta(record):
    img_h, img_w, img_c = record.img.shape

    return {
        # TODO: height and width from sample should be before padding
        # "img_shape": (record.img_size.height, record.img_size.width, img_c),
        "img_shape": (img_h, img_w, img_c),
        "pad_shape": (img_h, img_w, img_c),
        "flip": False,  # TODO: is this correct?
        "scale_factor": 1.0,  # TODO: is scale factor correct?
    }


def build_train_batch(records: Sequence[BaseRecord]):
    # tensor_images, tensor_masks = [], []
    # for record in records:
    #     # can be optimzed to be converted to tensor once at the end
    #     tensor_images.append(im2tensor(record.img))
    #     tensor_masks.append(
    #         tensor(record.segmentation.mask_array.data).long().squeeze()
    #     )

    # tensor_images = torch.stack(tensor_images)
    # tensor_masks = torch.stack(tensor_masks)

    # return (tensor_images, tensor_masks), records

    images, labels, bboxes, masks, img_metas = [], [], [], [], []
    for record in records:
        images.append(im2tensor(record.img))
        img_metas.append(_img_meta_mask(record))
        # labels.append(_labels(record))
        # bboxes.append(_bboxes(record))
        masks.append(tensor(record.segmentation.mask_array.data).long())

    data = {
        "img": torch.stack(images),
        "img_metas": img_metas,
        # "gt_labels": labels,
        # "gt_bboxes": bboxes,
        "gt_semantic_seg": torch.stack(masks),
    }

    return data, records


def build_infer_batch(records: Sequence[BaseRecord]):
    tensor_images = []
    for record in records:
        tensor_images.append(im2tensor(record.img))

    tensor_images = torch.stack(tensor_images)
    return (tensor_images,), records
