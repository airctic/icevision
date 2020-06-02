__all__ = ["AnnotationParser2"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from .parser import *
from .mixins import *


# TODO: Implement idmap
class AnnotationParser2(Parser, ABC):
    def parse(self, show_pbar: bool = True):
        parse_funcs = self.collect_parse_funcs()
        get_imageid = parse_funcs.pop("imageid")
        records = defaultdict(lambda: {name: [] for name in parse_funcs})

        for sample in pbar(self, show_pbar):
            self.prepare(sample)
            imageid = get_imageid(sample)
            for name, func in parse_funcs.items():
                # TODO: Assume everything is returned in a list
                records[imageid][name].append(func(sample))
        return dict(records)
