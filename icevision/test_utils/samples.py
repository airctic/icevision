__all__ = [
    "annotations_dict",
    "sample_image_info_parser",
    "sample_annotation_parser",
    "sample_combined_parser",
    "sample_records",
    "sample_dataset",
]

import icevision
from icevision.imports import *
from icevision import *

try:
    source = Path(icevision.__file__).parent.parent / "samples"
    annotations_dict = json.loads((source / "annotations.json").read())
except FileNotFoundError:
    annotations_dict = None


def sample_image_info_parser():
    return parsers.COCOImageInfoParser(
        infos=annotations_dict["images"], img_dir=source / "images"
    )


def sample_annotation_parser():
    return parsers.COCOAnnotationParser(annotations=annotations_dict["annotations"])


def sample_combined_parser():
    return parsers.CombinedParser(
        sample_image_info_parser(), sample_annotation_parser()
    )


def sample_records():
    parser = sample_combined_parser()
    data_splitter = RandomSplitter([0.8, 0.2], seed=42)
    return parser.parse(data_splitter)


def sample_dataset():
    train_rs, valid_rs = sample_records()
    return Dataset(train_rs)
