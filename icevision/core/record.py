__all__ = ["BaseRecord", "autofix_records"]

from icevision.imports import *
from icevision.utils import *
from collections.abc import MutableMapping
from copy import copy
from .record_mixins import *


# TODO: MutableMapping because of backwards compatability
class BaseRecord(ImageidRecordMixin, SizeRecordMixin, RecordMixin, MutableMapping):
    def autofix(self):
        # TODO: Check number of annotations is consistent (#bboxes==#labels==#masks)
        # checking number #masks is tricky, because single filepath can have multiple
        success_dict = self._autofix()
        success_list = np.array(list(success_dict.values()))
        keep_mask = reduce(np.logical_and, success_list)
        discard_idxs = np.where(keep_mask == False)[0]

        for i in discard_idxs:
            logger.info("Removed annotation with index: {}", i)
            self.remove_annotation(i)

        return success_dict

    def remove_annotation(self, i):
        # TODO: remove_annotation might work incorrectly with masks
        self._remove_annotation(i)

    def copy(self) -> "BaseRecord":
        return copy(self)

    def load(self) -> "BaseRecord":
        record = copy(self)
        record._load()
        return record

    # backwards compatiblity: implemented method to behave like a dict
    def __getitem__(self, key):
        return self.as_dict()[key]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __iter__(self):
        yield from self.as_dict()

    def __len__(self):
        return len(self.as_dict())


def autofix_records(records: Sequence[BaseRecord]) -> Sequence[BaseRecord]:
    for record in records:

        def _pre_replay():
            logger.info("Autofixing record with imageid: {}", record.imageid)

        with ReplaySink(_pre_replay) as sink:
            record.autofix()

    return records
