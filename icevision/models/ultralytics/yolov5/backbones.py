__all__ = [
    "small",
    "medium",
    "large",
    "extra_large",
]

from icevision.models.ultralytics.yolov5.utils import *


small = YoloV5BackboneConfig(model_name="yolov5s")

medium = YoloV5BackboneConfig(model_name="yolov5m")

large = YoloV5BackboneConfig(model_name="yolov5l")

extra_large = YoloV5BackboneConfig(model_name="yolov5x")
