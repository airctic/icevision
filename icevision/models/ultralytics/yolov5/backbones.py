__all__ = [
    "small",
    "small_p6",
    "medium",
    "medium_p6",
    "large",
    "large_p6",
    "extra_large",
    "extra_large_p6",
]

from icevision.models.ultralytics.yolov5.utils import *


small = YoloV5BackboneConfig(model_name="yolov5s")
small_p6 = YoloV5BackboneConfig(model_name="yolov5s6")

medium = YoloV5BackboneConfig(model_name="yolov5m")
medium_p6 = YoloV5BackboneConfig(model_name="yolov5m6")

large = YoloV5BackboneConfig(model_name="yolov5l")
large_p6 = YoloV5BackboneConfig(model_name="yolov5l6")

extra_large = YoloV5BackboneConfig(model_name="yolov5x")
extra_large_p6 = YoloV5BackboneConfig(model_name="yolov5x6")
