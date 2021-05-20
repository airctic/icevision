from icevision.models.fastai.unet import backbones
from icevision.models.fastai.unet.dataloaders import *
from icevision.models.fastai.unet.model import *

from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai:
    from icevision.models.fastai.unet import fastai
