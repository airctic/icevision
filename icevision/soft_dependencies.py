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
        self.fastai = soft_import("fastai")
        self.pytorch_lightning = soft_import("pytorch_lightning")
        self.effdet = soft_import("effdet")
        self.wandb = soft_import("wandb")
        self.resnest = soft_import("resnest")
        self.mmdet = soft_import("mmdet")
        self.yolov5 = soft_import("yolov5")
        self.sklearn = soft_import("sklearn")
        self.mmseg = soft_import("mmseg")
        self.sahi = soft_import("sahi")
        self.fiftyone = soft_import("fiftyone")
        self.pydicom = soft_import("pydicom")

    def check(self) -> Dict[str, bool]:
        return self.__dict__.copy()


SoftDependencies = _SoftDependencies()
