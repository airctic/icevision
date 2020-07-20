__all__ = [
    "filter_params",
    "unfreeze",
    "freeze",
    "transform_dataloader",
    "common_build_batch",
]

from mantisshrimp.imports import *
from mantisshrimp.parsers import *


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


def transform_dataloader(dataset, build_batch, batch_tfms=None, **dataloader_kwargs):
    collate_fn = partial(build_batch, batch_tfms=batch_tfms)
    return DataLoader(dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs)


def common_build_batch(records: Sequence[RecordType], batch_tfms=None):
    if batch_tfms is not None:
        records = batch_tfms(records)

    return records
