__all__ = ["load_txt", "create_tmp_dir", "mkdir", "extract_files"]

import numpy as np
import shutil
from pathlib import Path
from .utils import pbar


def load_txt(file):
    return np.loadtxt(file, dtype=str, delimiter="\n").tolist()


def create_tmp_dir(name: str, overwrite: bool = True) -> Path:
    path = Path("/tmp") / name

    if path.exists() and overwrite:
        shutil.rmtree(path)

    path.mkdir(parents=True)

    return path


def mkdir(path, exist_ok=False, parents=False, overwrite=False) -> Path:
    path = Path(path)

    if path.exists() and overwrite:
        shutil.rmtree(path)

    path.mkdir(exist_ok=exist_ok, parents=parents)
    return path


def extract_files(files, extract_to_dir, show_pbar: bool = True):
    for file in pbar(files, show=show_pbar):
        extract_path = extract_to_dir / Path(file).with_suffix("").name
        shutil.unpack_archive(file, extract_path)

    return extract_to_dir
