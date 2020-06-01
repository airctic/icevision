__all__ = ["DefaultImageInfoParser"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from .parser import *
from .mixins import *


class ImageInfoParser(Parser, ABC):
    def parse(self, show_pbar: bool = True):
        records = []
        for sample in pbar(self, show_pbar):
            self.prepare(sample)
            out = {name: func(sample) for name, func in self.parse_funcs.items()}
            records.append(out)
        return records


class DefaultImageInfoParser(
    ImageInfoParser,
    ImageidParserMixin,
    FilepathParserMixin,
    SizeParserMixin,
    SplitParserMixin,
    ABC,
):
    pass
