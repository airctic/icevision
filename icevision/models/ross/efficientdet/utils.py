__all__ = [
    "EfficientDetBackboneConfig",
]

from icevision.imports import *


@dataclass
class EfficientDetBackboneConfig:
    model_name: str
