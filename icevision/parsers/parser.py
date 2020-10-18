__all__ = ["ParserInterface", "Parser"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.parsers.mixins import *


# TODO: Rename to BaseParser
class ParserInterface(ABC):
    @abstractmethod
    def parse(
        self, data_splitter: DataSplitter, autofix: bool = True, show_pbar: bool = True
    ) -> List[List[RecordType]]:
        pass


class Parser(ImageidMixin, SizeMixin, ParserInterface, ABC):
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

    @abstractmethod
    def __iter__(self) -> Any:
        pass

    def prepare(self, o):
        pass

    def record_class(self) -> BaseRecord:
        return create_mixed_record(self.record_mixins())

    def parse_dicted(
        self, idmap: IDMap, show_pbar: bool = True
    ) -> Dict[int, RecordType]:

        Record = self.record_class()
        records = {}

        for sample in pbar(self, show_pbar):
            try:
                self.prepare(sample)
                true_imageid = self.imageid(sample)
                imageid = idmap[true_imageid]

                try:
                    record = records[imageid]
                except KeyError:
                    record = Record()

                self.parse_fields(sample, record)

                # HACK: fix imageid (needs to be transformed with idmap)
                record.set_imageid(imageid)
                records[imageid] = record

            except AbortParseRecord as e:
                logger.warning(
                    "Record with imageid: {} was skipped because: {}",
                    true_imageid,
                    str(e),
                )

        return dict(records)

    def parse(
        self,
        data_splitter: DataSplitter = None,
        idmap: IDMap = None,
        autofix: bool = True,
        show_pbar: bool = True,
    ) -> List[List[BaseRecord]]:
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
        all_splits_records = []
        for ids in splits:
            split_records = [records[i] for i in ids if i in records]

            if autofix:
                split_records = autofix_records(split_records)

            all_splits_records.append(split_records)

        return all_splits_records

    @classmethod
    def _templates(cls) -> List[str]:
        templates = super()._templates()
        return ["def __iter__(self) -> Any:"] + templates

    @classmethod
    def generate_template(cls):
        for template in cls._templates():
            print(f"{template}")
