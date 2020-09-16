__all__ = ["ParserInterface", "Parser"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.parsers.mixins import *


class ParserInterface(ABC):
    @abstractmethod
    def parse(
        self, data_splitter: DataSplitter, show_pbar: bool = True
    ) -> List[List[RecordType]]:
        pass


class Parser(ImageidMixin, ParserInterface, ABC):
    """Base class for all parsers, implements the main parsing logic.

    The actual fields to be parsed are defined by the mixins used when
    defining a custom parser. The only required field for all parsers
    is the `image_id`.

    # Examples

    Create a parser for image filepaths.
    ```python
    class FilepathParser(Parser, FilepathParserMixin):
        # implement required abstract methods
    ```
    """

    def prepare(self, o):
        pass

    @abstractmethod
    def __iter__(self) -> Any:
        pass

    def parse_dicted(
        self, idmap: IDMap, show_pbar: bool = True
    ) -> Dict[int, RecordType]:
        info_parse_funcs = self.collect_info_parse_funcs()
        annotation_parse_funcs = self.collect_annotation_parse_funcs()

        get_imageid = info_parse_funcs.pop("imageid")

        records = defaultdict(lambda: {name: [] for name in annotation_parse_funcs})
        for sample in pbar(self, show_pbar):
            self.prepare(sample)
            imageid = idmap[get_imageid(sample)]

            for name, func in info_parse_funcs.items():
                records[imageid][name] = func(sample)

            for name, func in annotation_parse_funcs.items():
                records[imageid][name].extend(func(sample))

        # check that all annotations have the same length
        # HACK: Masks is not checked, because it can be a single file with multiple masks
        annotations_names = [n for n in annotation_parse_funcs.keys() if n != "masks"]
        for imageid, record_annotations in records.items():
            record_annotations_len = {
                name: len(record_annotations[name]) for name in annotations_names
            }
            if not allequal(list(record_annotations_len.values())):
                true_imageid = idmap.get_id(imageid)
                # TODO: instead of immediatily raising the error, store the
                # result and raise at the end of the for loop for all records
                raise RuntimeError(
                    f"imageid->{true_imageid} has an inconsistent number of annotations"
                    f", all annotations must have the same length."
                    f"\nNumber of annotations: {record_annotations_len}"
                )

        return dict(records)

    def parse(
        self,
        data_splitter: DataSplitter = None,
        idmap: IDMap = None,
        show_pbar: bool = True,
    ) -> List[List[RecordType]]:
        """Loops through all data points parsing the required fields.

        # Arguments
            data_splitter: How to split the parsed data, defaults to a [0.8, 0.2] random split.
            idmap: Maps from filenames to unique ids, pass an `IDMap()` if you need this information.
            show_pbar: Whether or not to show a progress bar while parsing the data.

        # Returns
            A list of records for each split defined by `data_splitter`.
        """
        idmap = idmap or IDMap()
        data_splitter = data_splitter or RandomSplitter([0.8, 0.2])
        records = self.parse_dicted(show_pbar=show_pbar, idmap=idmap)
        splits = data_splitter(idmap=idmap)
        return [[{"imageid": id, **records[id]} for id in ids] for ids in splits]

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return ["def __iter__(self) -> Any:"] + templates

    @classmethod
    def generate_template(cls):
        for template in cls._templates():
            print(f"{template}")
