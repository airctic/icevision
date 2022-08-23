import sys, os, re, shutil, typing, itertools, operator, math, warnings, json, random
import functools, io, cv2, mimetypes, torch, torchvision, dataclasses, zipfile, pickle
import PIL
import rasterio
from copy import copy, deepcopy

from pdb import set_trace
from dataclasses import dataclass
from fastcore.foundation import *
from fastcore.basics import *

import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    MethodType,
    FunctionType,
    SimpleNamespace,
)

from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
    abstractclassmethod,
    abstractstaticmethod,
)
from pathlib import Path
from collections import defaultdict, OrderedDict, namedtuple
from enum import Enum
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils
from contextlib import contextmanager
from typing import *
from operator import itemgetter, attrgetter
from functools import partial, reduce

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

from loguru import logger

# Soft imports
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai:
    import fastai.vision.all as fastai

if SoftDependencies.pytorch_lightning:
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers

if SoftDependencies.wandb:
    import wandb

if SoftDependencies.sklearn:
    import sklearn

if SoftDependencies.pydicom:
    import pydicom

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
