__all__ = ["get_data_dir"]

from icevision.imports import *

root_dir = Path.home() / ".icevision"
root_dir.mkdir(exist_ok=True)

data_dir = root_dir / "data"
data_dir.mkdir(exist_ok=True)


def get_data_dir():
    return data_dir
