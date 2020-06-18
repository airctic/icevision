__all__ = ["get_data_dir"]

from mantisshrimp.imports import *

root_dir = Path.home() / ".mantisshrimp"
root_dir.mkdir(exist_ok=True)

data_dir = root_dir / "data"
data_dir.mkdir(exist_ok=True)


def get_data_dir():
    return data_dir
