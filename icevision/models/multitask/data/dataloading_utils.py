"""
This may be a temporary file that may eventually be removed,
as it only slightly modifies an existing function.
"""

__all__ = ["unload_records", "assign_classification_targets_from_record"]


from icevision.core.record_type import RecordType
from typing import Any, Dict, Optional, Callable, Sequence, Tuple
from icevision.core.record_components import ClassificationLabelsRecordComponent


def unload_records(
    build_batch: Callable, build_batch_kwargs: Optional[Dict] = None
) -> Tuple[Tuple[Any, ...], Sequence[RecordType]]:
    """
    This decorator function unloads records to not carry them around after batch creation.
      It also optionally accepts `build_batch_kwargs` that are to be passed into
      `build_batch`. These aren't accepted as keyword arguments as those are reserved
      for PyTorch's DataLoader class which is used later in this chain of function calls

    Args:
        build_batch (Callable): A collate function that describes how to mash records
                                into a batch of inputs for a model
        build_batch_kwargs (Optional[Dict], optional): Keyword arguments to pass into
                                                       `build_batch`. Defaults to None.

    Returns:
        Tuple[Tuple[Any, ...], Sequence[RecordType]]: [description]
    """
    build_batch_kwargs = build_batch_kwargs or {}
    assert isinstance(build_batch_kwargs, dict)

    def inner(records):
        tupled_output, records = build_batch(records, **build_batch_kwargs)
        for record in records:
            record.unload()
        return tupled_output, records

    return inner


def assign_classification_targets_from_record(classification_labels: dict, record):
    for comp in record.components:
        name = comp.task.name
        if isinstance(comp, ClassificationLabelsRecordComponent):
            if comp.is_multilabel:
                labels = comp.one_hot_encoded()
                classification_labels[name].append(labels)
            else:
                labels = comp.label_ids
                classification_labels[name].extend(labels)
