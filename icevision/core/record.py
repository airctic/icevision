__all__ = ["BaseRecord", "autofix_records", "create_mixed_record"]

from icevision.imports import *
from icevision.utils import *
from collections.abc import MutableMapping
from copy import copy
from .record_mixins import *
from .exceptions import *


# TODO: MutableMapping because of backwards compatability
class BaseRecord(ImageidRecordMixin, SizeRecordMixin, RecordMixin, MutableMapping):
    def num_annotations(self) -> Dict[str, int]:
        return self._num_annotations()

    def check_num_annotations(self):
        num_annotations = self.num_annotations()
        if len(set(num_annotations.values())) > 1:
            msg = "\n".join([f"\t- {v} for {k}" for k, v in num_annotations.items()])
            raise AutofixAbort(
                "Number of items should be the same for each annotation type"
                f", but got:\n{msg}"
            )

    def autofix(self):
        self.check_num_annotations()

        success_dict = self._autofix()
        success_list = np.array(list(success_dict.values()))
        keep_mask = reduce(np.logical_and, success_list)
        discard_idxs = np.where(keep_mask == False)[0]

        for i in discard_idxs:
            logger.log(
                "AUTOFIX-REPORT",
                "Removed annotation with index: {}, "
                "for more info check the AUTOFIX-FAIL messages above",
                i,
            )
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

    def __repr__(self) -> str:
        _reprs = self._repr()
        _repr = "".join(f"\n\t- {o}" for o in _reprs)
        return f"Record:{_repr}"

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
    keep_records = []
    for record in records:

        def _pre_replay():
            logger.log(
                "AUTOFIX-START",
                "️🔨  Autofixing record with imageid: {}  ️🔨",
                record.imageid,
            )

        with ReplaySink(_pre_replay) as sink:
            try:
                record.autofix()
                keep_records.append(record)
            except AutofixAbort as e:
                logger.warning(
                    "🚫 Record could not be autofixed and will be removed because: {}",
                    str(e),
                )

    return keep_records


def create_mixed_record(
    mixins: Sequence[Type[RecordMixin]], add_base: bool = True
) -> Type[BaseRecord]:
    mixins = (BaseRecord, *mixins) if add_base else tuple(mixins)

    TemporaryRecord = type("Record", mixins, {})
    class_name = "".join([o.__name__ for o in TemporaryRecord.mro()])

    Record = type(class_name, mixins, {})
    return patch_class_to_main(Record)
