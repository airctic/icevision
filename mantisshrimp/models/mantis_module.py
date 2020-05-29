__all__ = ["MantisModule"]

from ..imports import *
from .parameters_splits_module_mixin import *


class MantisModule(ParametersSplitsModuleMixin, LightningModule, ABC):
    pass
