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
        records = {name: defaultdict(list) for name in parse_funcs}
        records["imageid"] = set()

        # TODO: Assume everything is returned in a list
        for sample in pbar(self, show_pbar):
            self.prepare(sample)
            imageid = get_imageid(sample)
            records["imageid"].add(imageid)
            for name, func in parse_funcs.items():
                records[name][imageid].append(func(sample))
        return records
