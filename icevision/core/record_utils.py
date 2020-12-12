__all__ = ["aggregate_records_objects"]

from icevision.imports import *
from .record import BaseRecord


def aggregate_records_objects(
    records: List[BaseRecord],
) -> Dict[str, List[Dict[str, Any]]]:
    objects = defaultdict(list)
    for record in records:
        for k, v in record.aggregate_objects().items():
            objects[k].extend(v)
    return dict(objects)
