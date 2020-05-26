__all__ = ['MantisModule']

from ..imports import *
from .prepare_optimizer_module_mixin import *

class MantisModule(PrepareOptimizerModuleMixin, LightningModule, ABC):
    pass