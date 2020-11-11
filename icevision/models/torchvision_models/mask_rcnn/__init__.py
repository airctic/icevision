# import icevision.models.torchvision_models.backbones as backbones
from icevision.models.torchvision_models import backbones
from icevision.models.torchvision_models.loss_fn import *

from icevision.models.torchvision_models.mask_rcnn.dataloaders import *
from icevision.models.torchvision_models.mask_rcnn.model import *
from icevision.models.torchvision_models.mask_rcnn.prediction import *
from icevision.models.torchvision_models.mask_rcnn.show_results import *

# Soft dependencies
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai:
    import icevision.models.torchvision_models.mask_rcnn.fastai

if SoftDependencies.pytorch_lightning:
    import icevision.models.torchvision_models.mask_rcnn.lightning
