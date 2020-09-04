__all__ = ["load", "class_map"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *


def load(force_download: bool = False) -> Path:
    images_url = "https://drive.google.com/uc?id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx&export=download"
    lists_url = "https://drive.google.com/uc?export=download&id=1vZuZPqha0JjmwkdaS_XtYryE3Jf5Q1AC"
    annotations_url = "https://drive.google.com/uc?export=download&id=16NsbTpMs5L6hT4hUJAmpW2u7wH326WTR"
    data_dir = get_data_dir() / "birds"
    data_dir.mkdir(exist_ok=True, parents=True)

    if not data_dir.exists() or force_download:
        download_and_extract_gdrive(
            url=images_url, filename="images.tgz", extract_dir=data_dir
        )
        download_and_extract_gdrive(
            url=lists_url, filename="lists.tgz", extract_dir=data_dir
        )
        download_and_extract_gdrive(
            url=annotations_url, filename="annotations.tgz", extract_dir=data_dir
        )

    return data_dir


def class_map(data_dir: Union[str, Path], background: Optional[int] = 0) -> ClassMap:
    classes_filepath = Path(data_dir) / "lists/classes.txt"
    classes = classes_filepath.read().splitlines()
    return ClassMap(classes, background=background)
