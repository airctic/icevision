import cv2, json, torch, torchvision, dataclasses, zipfile

from fastcore.imports import *
from fastcore.foundation import *
from fastcore.utils import *

import pandas as pd
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
