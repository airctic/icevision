__all__ = ["mobilenet"]

from icevision.backbones import mobilenet as mobilenet_fn
from icevision.models.torchvision.backbones.backbone_config import (
    TorchvisionBackboneConfig,
)

mobilenet = TorchvisionBackboneConfig(mobilenet_fn)
