__all__ = ["ParserInterface", "Parser"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from .mixins import *
from .splits import *


class ParserInterface(ABC):
    @abstractmethod
    def parse(
        self, data_splitter: DataSplitter, show_pbar: bool = True
    ) -> List[List[dict]]:
        pass


class Parser(ImageidParserMixin, ParserInterface, ABC):
    def prepare(self, o):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def parse_dicted(self, show_pbar: bool = True) -> Dict[int, dict]:
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

    def parse(
        self, data_splitter: DataSplitter = None, show_pbar: bool = True
    ) -> List[List[dict]]:
        data_splitter = data_splitter or SingleSplitSplitter()
        records = self.parse_dicted(show_pbar=show_pbar)
        splits = data_splitter(records.keys())
        return [[{"imageid": id, **records[id]} for id in ids] for ids in splits]
