__all__ = ["aggregate_records_objects"]

from icevision.imports import *
from .record import BaseRecord


def aggregate_records_objects(
    records: List[BaseRecord],
) -> Dict[str, List[Dict[str, Any]]]:
    objects = defaultdict(lambda: defaultdict(list))
    for record in records:
        for composite_name, composite_dict in record.aggregate_objects().items():
            for component_name, component_value in composite_dict.items():
                objects[composite_name][component_name].extend(component_value)
    return dict(objects)
