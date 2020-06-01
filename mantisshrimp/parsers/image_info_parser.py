__all__ = ["DefaultImageInfoParser"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from .mixins import *


class Parser(ImageidParserMixin, ABC):
    def __init__(self):
        self.parse_fns = self.collect_parse_funcs()

    def prepare(self, o):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def parse(self, show_pbar: bool = True):
        records = []
        for sample in pbar(self, show_pbar):
            self.prepare(sample)
            out = {name: fn(sample) for name, fn in self.parse_fns.items()}
            records.append(out)
        return records


class DefaultImageInfoParser(
    Parser,
    ImageidParserMixin,
    FilepathParserMixin,
    SizeParserMixin,
    SplitParserMixin,
    ABC,
):
    pass
