from icevision.models.yolo.yolov5.dataloaders import *
from icevision.models.yolo.yolov5.model import *

# Soft dependencies
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai:
    from icevision.models.yolo.yolov5 import fastai
