__all__ = ["DataPreparer", "DefaultDataPreparer"]

from mantisshrimp.data_preparer.mixins import *


class DataPreparer(PreparerMixin):
    def __call__(self, record):
        return self.prepare(record)

    def prepare(self, record):
        record = record.copy()
        super().prepare(record)
        return record


class DefaultDataPreparer(DataPreparer, ImgPreparerMixin, MaskPreparerMixin):
    pass
