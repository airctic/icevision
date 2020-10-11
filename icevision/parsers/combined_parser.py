__all__ = ["CombinedParser"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.parsers.parser import *


class CombinedParser(ParserInterface):
    def __init__(self, *parsers: Parser):
        self.parsers = parsers

    def record_class(self) -> BaseRecord:
        record_bases = [parser.record_class() for parser in self.parsers]
        return type("Record", record_bases, {})

    def parse(
        self,
        data_splitter=None,
        idmap: IDMap = None,
        autofix: bool = True,
        show_pbar: bool = True,
    ):
        data_splitter = data_splitter or RandomSplitter([0.8, 0.2])
        idmap = idmap or IDMap()

        parsers_records = [
            o.parse_dicted(idmap=idmap, show_pbar=show_pbar) for o in self.parsers
        ]

        ids = [set(o.keys()) for o in parsers_records]
        valid_ids = set.intersection(*ids)
        excluded = set.union(*ids) - valid_ids
        logger.info(f"Removed {len(excluded)} files: {excluded}")

        intersect_idmap = idmap.filter_ids(ids=valid_ids)
        splits = data_splitter(idmap=intersect_idmap)

        # TODO: Confusing names
        # combine keys for all parsers
        Record = self.record_class()
        split_records = []
        for ids in splits:
            current_records = []
            for id in ids:
                record = Record()
                for records in parsers_records:
                    # HACK: directly updating __dict__
                    record.__dict__.update(records[id].as_dict())
                current_records.append(record)

            if autofix:
                autofix_records(current_records)

            split_records.append(current_records)

        return split_records
