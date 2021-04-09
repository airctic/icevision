from icevision.models.torchvision.loss_fn import *

from icevision.models.torchvision.faster_rcnn import backbones
from icevision.models.torchvision.faster_rcnn.dataloaders import *
from icevision.models.torchvision.faster_rcnn.model import *
from icevision.models.torchvision.faster_rcnn.prediction import *
from icevision.models.torchvision.faster_rcnn.show_batch import *
from icevision.models.torchvision.faster_rcnn.show_results import *


# Soft dependencies
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai:
    import icevision.models.torchvision.faster_rcnn.fastai

if SoftDependencies.pytorch_lightning:
    import icevision.models.torchvision.faster_rcnn.lightning
