__all__ = ["MantisModule"]

from ..imports import *
from .prepare_optimizer_module_mixin import *
from .parameters_splits_module_mixin import *


class MantisModule(
    PrepareOptimizerModuleMixin, ParametersSplitsModuleMixin, LightningModule, ABC
):
    pass
