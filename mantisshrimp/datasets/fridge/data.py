__all__ = ["class_map", "load"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *

_CLASSES = sorted({"milk_bottle", "carton", "can", "water_bottle"})


def class_map(background: Optional[int] = 0) -> ClassMap:
    return ClassMap(classes=_CLASSES, background=background)


def load(force_download=False):
    base_url = "https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip"
    save_dir = get_data_dir() / "fridge"
    save_dir.mkdir(exist_ok=True)

    save_path = save_dir / "data.zip"
    if not save_path.exists() or force_download:
        download_and_extract(url=base_url, save_path=save_path)

    return save_dir
