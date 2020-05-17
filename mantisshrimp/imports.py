import cv2,json,torch,torchvision,dataclasses
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning.loggers as loggers

from enum import Enum
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils
from contextlib import contextmanager
from typing import *

from torch import tensor, Tensor
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam,AdamW,Adagrad,Adadelta,RMSprop
from torch.optim.lr_scheduler import (LambdaLR,StepLR,MultiStepLR,MultiplicativeLR,OneCycleLR,
                                      CosineAnnealingLR, CosineAnnealingWarmRestarts)

from torchvision.transforms.functional import to_tensor as im2tensor
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.faster_rcnn import FasterRCNN,FastRCNNPredictor,fasterrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN,MaskRCNNPredictor,maskrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN,KeypointRCNNPredictor,keypointrcnn_resnet50_fpn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.trainer.distrib_parts import get_all_available_gpus
from pytorch_lightning.callbacks import LearningRateLogger

from fastcore.imports import *
from fastcore.test import *
from fastcore.foundation import *
from fastcore.utils import *

try: import albumentations as A
except ModuleNotFoundError: pass
