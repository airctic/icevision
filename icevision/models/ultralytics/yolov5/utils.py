__all__ = [
    "YoloV5BackboneConfig",
]

from icevision.imports import *
from icevision.backbones import BackboneConfig


class YoloV5BackboneConfig(BackboneConfig):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pretrained: bool

    def __call__(self, pretrained: bool = True) -> "YoloV5BackboneConfig":
        """Completes the configuration of the backbone

        # Arguments
            pretrained: If True, use a pretrained backbone (on COCO).
        """
        self.pretrained = pretrained
        return self
