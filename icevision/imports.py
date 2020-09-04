import cv2, json, torch, torchvision, dataclasses, zipfile

from fastcore.imports import *
from fastcore.foundation import *
from fastcore.utils import *

import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
    abstractclassmethod,
    abstractstaticmethod,
)
from collections import defaultdict, OrderedDict
from enum import Enum
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils
from contextlib import contextmanager
from typing import *

from torch import tensor, Tensor
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW, Adagrad, Adadelta, RMSprop
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    MultiStepLR,
    MultiplicativeLR,
    OneCycleLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

from torchvision.transforms.functional import to_tensor as im2tensor

# Soft imports
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai2:
    import fastai2.vision.all as fastai

if SoftDependencies.pytorch_lightning:
    import pytorch_lightning as pl


# TODO: Stop importing partial from fastcore and move this to utils
class partial:
    """Wraps functools.partial, same functionality.

    Modifies the original partial `__repr__` and `__str__` in other to fix #270
    """

    def __init__(self, func, *args, **kwargs):
        self._partial = functools.partial(func, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._partial(*args, **kwargs)

    def __str__(self):
        name = self._partial.func.__name__
        partial_str = str(self._partial)
        return re.sub(r"<.+>", name, partial_str)

    def __repr__(self):
        return str(self)
