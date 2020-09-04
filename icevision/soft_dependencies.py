__all__ = ["soft_import", "SoftDependencies"]

import importlib
from typing import *


def soft_import(name: str):
    try:
        importlib.import_module(name)
        return True
    except ModuleNotFoundError as e:
        if str(e) != f"No module named '{name}'":
            raise e
        return False


class _SoftDependencies:
    def __init__(self):
        self.fastai2 = soft_import("fastai2")
        self.pytorch_lightning = soft_import("pytorch_lightning")
        self.albumentations = soft_import("albumentations")
        self.effdet = soft_import("effdet")

    def check(self) -> Dict[str, bool]:
        return self.__dict__.copy()


SoftDependencies = _SoftDependencies()
