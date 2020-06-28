__all__ = ["CombinedParser"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.splits import *


class CombinedParser(ParserInterface):
    def __init__(self, *parsers: Parser):
        self.parsers = parsers

    def parse(self, data_splitter=None, idmap: IDMap = None, show_pbar: bool = True):
        data_splitter = data_splitter or SingleSplitSplitter()
        idmap = idmap or IDMap()

        parsers_records = [
            o.parse_dicted(idmap=idmap, show_pbar=show_pbar) for o in self.parsers
        ]

        ids = [set(o.keys()) for o in parsers_records]
        valid_ids = set.intersection(*ids)
        excluded = set.union(*ids) - valid_ids
        print(f"Removed {len(excluded)} files: {excluded}")

        splits = data_splitter(valid_ids)

        # TODO: Confusing names
        # combine keys for all parsers
        split_records = []
        for ids in splits:
            current_records = []
            for id in ids:
                record = {"imageid": id}
                for records in parsers_records:
                    record.update(records[id])
                current_records.append(record)
            split_records.append(current_records)
        return split_records
