__all__ = ["Parser"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from .mixins import *


class Parser(ImageidParserMixin, ABC):
    def prepare(self, o):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def parse(self, show_pbar: bool = True):
        info_parse_funcs = self.collect_info_parse_funcs()
        annotation_parse_funcs = self.collect_annotation_parse_funcs()
        get_imageid = info_parse_funcs.pop("imageid")
        records = defaultdict(lambda: {name: [] for name in annotation_parse_funcs})

        for sample in pbar(self, show_pbar):
            self.prepare(sample)
            imageid = get_imageid(sample)
            for name, func in info_parse_funcs.items():
                records[imageid][name] = func(sample)
            for name, func in annotation_parse_funcs.items():
                # TODO: Assume everything is returned in a list
                records[imageid][name].append(func(sample))
        return dict(records)
