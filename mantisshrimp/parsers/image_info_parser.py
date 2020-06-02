__all__ = ["DefaultImageInfoParser"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from .parser import *
from .mixins import *


class ImageInfoParser(Parser, ABC):
    def parse(self, show_pbar: bool = True):
        parse_funcs = self.collect_parse_funcs()
        get_imageid = parse_funcs.pop("imageid")
        records = {}
        for sample in pbar(self, show_pbar):
            self.prepare(sample)
            imageid = get_imageid(sample)
            records[imageid] = {
                name: func(sample) for name, func in parse_funcs.items()
            }
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
