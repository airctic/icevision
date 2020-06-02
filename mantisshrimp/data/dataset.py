__all__ = ["Dataset"]

from ..imports import *
from ..utils import *
from ..core import *
from ..transforms import *
from mantisshrimp.data_preparer import *


class Dataset:
    def __init__(
        self,
        records: List[dict],
        tfm: Transform = None,
        data_preparer: DataPreparer = None,
    ):
        self.records = records
        self.tfm = tfm
        self.data_preparer = data_preparer or DefaultDataPreparer()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self._getitem(i=i)

    def _getitem(self, i):
        data = self.data_preparer(self.records[i])
        if self.tfm is not None:
            data = self.tfm(data)
        return data

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items>"
