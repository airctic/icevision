from icevision.models.utils import *

from icevision.models.torchvision_models import faster_rcnn, mask_rcnn, retinanet

# Soft dependencies
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.effdet:
    from icevision.models import efficientdet
