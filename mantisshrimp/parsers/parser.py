__all__ = ["Parser"]

from mantisshrimp.imports import *
from .mixins import *


class Parser(ImageidParserMixin, ABC):
    def __init__(self):
        self.parse_funcs = self.collect_parse_funcs()

    def prepare(self, o):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def parse(self, show_pbar: bool = True):
        pass
