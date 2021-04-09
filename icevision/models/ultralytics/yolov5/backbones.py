__all__ = [
    "small",
    "medium",
    "large",
]

from icevision.models.ultralytics.yolov5.utils import *


small = YoloV5BackboneConfig(model_name="yolov5s")

medium = YoloV5BackboneConfig(model_name="yolov5m")

large = YoloV5BackboneConfig(model_name="yolov5l")
