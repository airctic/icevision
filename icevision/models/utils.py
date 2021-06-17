__all__ = [
    "filter_params",
    "unfreeze",
    "freeze",
    "transform_dl",
    "apply_batch_tfms",
    "_predict_from_dl",
]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.parsers import *

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def filter_params(
    module: nn.Module, bn: bool = True, only_trainable=False
) -> Generator:
    """Yields the trainable parameters of a given module.

    Args:
        module: A given module
        bn: If False, don't return batch norm layers

    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not isinstance(module, BN_TYPES) or bn:
            for param in module.parameters():
                if not only_trainable or param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(
                module=child, bn=bn, only_trainable=only_trainable
            ):
                yield param


def unfreeze(params):
    for p in params:
        p.requires_grad = True


def freeze(params):
    for p in params:
        p.requires_grad = False


def transform_dl(
    dataset,
    build_batch,
    batch_tfms=None,
    build_batch_kwargs: dict = {},
    **dataloader_kwargs,
):
    """Creates collate_fn from build_batch (collate function) by decorating it with apply_batch_tfms and unload_records"""
    collate_fn = apply_batch_tfms(
        build_batch, batch_tfms=batch_tfms, **build_batch_kwargs
    )
    collate_fn = unload_records(collate_fn)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def apply_batch_tfms(build_batch, batch_tfms=None, **build_batch_kwargs):
    """This decorator function applies batch_tfms to records before passing them to build_batch"""

    def inner(records):
        if batch_tfms is not None:
            records = batch_tfms(records)
        return build_batch(records, **build_batch_kwargs)

    return inner


def unload_records(build_batch):
    """This decorator function unloads records to not carry them around after batch creation"""

    def inner(records):
        tupled_output, records = build_batch(records)
        for record in records:
            record.unload()
        return tupled_output, records

    return inner


@torch.no_grad()
def _predict_from_dl(
    predict_fn,
    model: nn.Module,
    infer_dl: DataLoader,
    keep_images: bool = False,
    show_pbar: bool = True,
    **predict_kwargs,
) -> List[Prediction]:
    all_preds = []
    for batch, records in pbar(infer_dl, show=show_pbar):
        preds = predict_fn(
            model=model,
            batch=batch,
            records=records,
            keep_images=keep_images,
            **predict_kwargs,
        )
        all_preds.extend(preds)

    return all_preds
