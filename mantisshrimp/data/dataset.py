__all__ = ["Dataset"]

from mantisshrimp.imports import *
from mantisshrimp.transforms import *
from mantisshrimp.data.prepare_record import *


class Dataset:
    def __init__(
        self, records: List[dict], tfm: Transform = None, prepare_record=None,
    ):
        self.records = records
        self.tfm = tfm
        self.prepare_record = prepare_record or default_prepare_record

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        data = self.prepare_record(self.records[i])
        if self.tfm is not None:
            data = self.tfm(data)
        return data

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items>"
