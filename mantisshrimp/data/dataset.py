__all__ = ['Dataset']

from ..utils import *
from ..core import *

class Dataset:
    def __init__(self, records, tfm=None): self.records, self.tfm = records, tfm

    def __len__(self): return len(self.records)

    def __getitem__(self, i): return self._getitem(i=i)

    def _getitem(self, i):
        item = Item.from_record(self.records[i])
        return self.tfm(item) if notnone(self.tfm) else item

    def __repr__(self): return f'<{self.__class__.__name__} with {len(self.records)} items>'
