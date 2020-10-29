__all__ = ["get_data_dir", "get_root_dir"]

from icevision.imports import *

root_dir = Path.home() / ".icevision"
root_dir.mkdir(exist_ok=True)

data_dir = root_dir / "data"
data_dir.mkdir(exist_ok=True)


def get_data_dir():
    return data_dir


def get_root_dir():
    return root_dir
