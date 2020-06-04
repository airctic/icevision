__all__ = ["CategoryParser"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *
from mantisshrimp.parsers.parser import *


class CategoryParser(Parser, ABC):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __iter__(self):
        yield from self.data

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def id(self, o):
        pass

    @abstractmethod
    def name(self, o):
        pass

    def parse_dicted(self, show_pbar=True):
        return CategoryMap(
            [
                Category(self.id(sample), self.name(sample))
                for sample in pbar(self, show=show_pbar)
            ]
        )
