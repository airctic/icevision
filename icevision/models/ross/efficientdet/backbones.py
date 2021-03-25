__all__ = [
    "tf_lite0",
    "d0",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
    "d7x",
]

from icevision.models.ross.efficientdet.utils import *


tf_lite0 = EfficientDetBackboneConfig(model_name="tf_efficientdet_lite0")

d0 = EfficientDetBackboneConfig(model_name="efficientdet_d0")

d1 = EfficientDetBackboneConfig(model_name="efficientdet_d1")

d2 = EfficientDetBackboneConfig(model_name="efficientdet_d2")

d3 = EfficientDetBackboneConfig(model_name="efficientdet_d3")

d4 = EfficientDetBackboneConfig(model_name="efficientdet_d4")

d5 = EfficientDetBackboneConfig(model_name="efficientdet_d5")

d6 = EfficientDetBackboneConfig(model_name="efficientdet_d6")

d7 = EfficientDetBackboneConfig(model_name="efficientdet_d7")

d7x = EfficientDetBackboneConfig(model_name="efficientdet_d7x")
