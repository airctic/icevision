from mantisshrimp.imports import *
from mantisshrimp import *


def plot_size_histogram(records: List[RecordType]):
    height_sum = 0
    width_sum = 0
    for record in records:
        height_sum += record["height"]
        width_sum += record["width"]

    height_mean = height_sum / len(records)
    width_mean = width_sum / len(records)
