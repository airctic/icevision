"""
This may be a temporary file that may eventually be removed,
as it only slightly modifies an existing function.
"""

__all__ = ["unload_records"]


from typing import Dict, Optional, Callable

from numpy.lib.arraysetops import isin


def unload_records(build_batch: Callable, build_batch_kwargs: Optional[Dict] = None):
    """
    This decorator function unloads records to not carry them around after batch creation
      and will also accept any additional args required by the `build_batch`` function
    """
    build_batch_kwargs = build_batch_kwargs or {}
    assert isinstance(build_batch_kwargs, dict)

    def inner(records):
        tupled_output, records = build_batch(records, **build_batch_kwargs)
        for record in records:
            record.unload()
        return tupled_output, records

    return inner
