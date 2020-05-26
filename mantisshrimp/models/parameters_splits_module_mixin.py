__all__ = ["ParametersSplitsModuleMixin"]

from ..imports import *
from .utils import *


class ParametersSplitsModuleMixin(LightningModule, ABC):
    # TODO: Implement default splitter
    @abstractmethod
    def model_splits(self):
        pass

    def params_splits(self, only_trainable=False):
        """ Get parameters from model splits
        """
        for split in self.model_splits():
            params = list(filter_params(split, only_trainable=only_trainable))
            if params:
                yield params

    def trainable_params_splits(self):
        """ Get trainable parameters from model splits
            If a parameter group does not have trainable params, it does not get added
        """
        return self.params_splits(only_trainable=True)
