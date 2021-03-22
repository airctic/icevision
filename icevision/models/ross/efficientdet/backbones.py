__all__ = [
    "tf_efficientdet_lite0",
    "efficientdet_d0",
    "efficientdet_d1",
    "efficientdet_d2",
    "efficientdet_d3",
    "efficientdet_d4",
    "efficientdet_d5",
    "efficientdet_d6",
    "efficientdet_d7",
    "efficientdet_d7x",
]

from icevision.models.ross.efficientdet.utils import *


tf_efficientdet_lite0 = EfficientDetBackboneConfig(model_name="tf_efficientdet_lite0")

efficientdet_d0 = EfficientDetBackboneConfig(model_name="efficientdet_d0")

efficientdet_d1 = EfficientDetBackboneConfig(model_name="efficientdet_d1")

efficientdet_d2 = EfficientDetBackboneConfig(model_name="efficientdet_d2")

efficientdet_d3 = EfficientDetBackboneConfig(model_name="efficientdet_d3")

efficientdet_d4 = EfficientDetBackboneConfig(model_name="efficientdet_d4")

efficientdet_d5 = EfficientDetBackboneConfig(model_name="efficientdet_d5")

efficientdet_d6 = EfficientDetBackboneConfig(model_name="efficientdet_d6")

efficientdet_d7 = EfficientDetBackboneConfig(model_name="efficientdet_d7")

efficientdet_d7x = EfficientDetBackboneConfig(model_name="efficientdet_d7x")
