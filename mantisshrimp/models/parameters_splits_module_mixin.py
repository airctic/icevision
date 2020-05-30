__all__ = ["ParametersSplitsModuleMixin"]

from ..imports import *
from .utils import *


class ParametersSplitsModuleMixin(LightningModule, ABC):
    def model_splits(self):
        return self.children()

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

    def freeze_to(self, n: int = None):
        """ Freezes model until certain layer
        """
        unfreeze(self.parameters())
        for params in list(self.params_splits())[:n]:
            freeze(params)

    def get_optimizer_param_groups(self, lr=0):
        return [{"params": params, "lr": lr} for params in self.params_splits()]

    #
    # def get_lrs(self, lr):
    #     n_splits = len(list(self.params_splits()))
    #     if isinstance(lr, numbers.Number):
    #         return [lr] * n_splits
    #     if isinstance(lr, (tuple, list)):
    #         assert len(lr) == len(list(self.params_splits()))
    #         return lr
    #     raise ValueError
