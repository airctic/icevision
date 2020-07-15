__all__ = ["CLASSES", "load"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *

CLASSES = {
    "milk_bottle",
    "carton",
    "can",
    "water_bottle",
}
CLASSES = sorted(CLASSES)
CLASSES = ["background"] + CLASSES


def load(force_download=False):
    base_url = "https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip"
    save_dir = get_data_dir() / "fridge"
    save_dir.mkdir(exist_ok=True)

    save_path = save_dir / "data.zip"
    if not save_path.exists() or force_download:
        download_and_extract(url=base_url, save_path=save_path)

    return save_dir
