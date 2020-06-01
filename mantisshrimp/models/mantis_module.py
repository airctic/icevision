__all__ = ["MantisModule"]

from ..imports import *
from .parameters_splits_module_mixin import *


class MantisModule(ParametersSplitsModuleMixin, LightningModule, ABC):
    @classmethod
    @abstractmethod
    def dataloader(cls, **kwargs) -> DataLoader:
        """ Returns the specific dataloader for this class
        """
