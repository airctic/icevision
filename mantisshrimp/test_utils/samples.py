__all__ = [
    "sample_image_info_parser",
    "sample_annotation_parser",
    "sample_combined_parser",
    "sample_records",
    "sample_dataset",
]

import mantisshrimp
from mantisshrimp.imports import *
from mantisshrimp import *

try:
    source = Path(mantisshrimp.__file__).parent.parent / "samples"
    annotations_dict = json.loads((source / "annotations.json").read())
except FileNotFoundError:
    pass


def sample_image_info_parser():
    return COCOImageInfoParser(annotations_dict["images"], source / "images")


def sample_annotation_parser():
    return COCOAnnotationParser2(annotations_dict["annotations"])


def sample_combined_parser():
    return CombinedParser(sample_image_info_parser(), sample_annotation_parser())


def sample_records():
    parser = sample_combined_parser()
    data_splitter = RandomSplitter([0.8, 0.2], seed=42)
    return parser.parse(data_splitter)


def sample_dataset():
    train_rs, valid_rs = sample_records()
    return Dataset(train_rs)
