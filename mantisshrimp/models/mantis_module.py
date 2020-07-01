__all__ = ["MantisModule"]

from mantisshrimp.imports import *
from mantisshrimp.models.device_module_mixin import *
from mantisshrimp.models.parameters_splits_module_mixin import *


class MantisModule(DeviceModuleMixin, ParametersSplitsModuleMixin, nn.Module, ABC):
    @abstractmethod
    def load_state_dict(
        self, state_dict: Dict[str, Tensor], strict: bool = True,
    ):
        """ Load weights
        """

    @classmethod
    @abstractmethod
    def dataloader(cls, **kwargs) -> DataLoader:
        """ Returns the specific dataloader for this class
        """
