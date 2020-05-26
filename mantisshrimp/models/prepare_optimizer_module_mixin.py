__all__ = ["PrepareOptimizerModuleMixin"]

from ..imports import *
from .utils import *
from .parameters_splits_module_mixin import ParametersSplitsModuleMixin


class PrepareOptimizerModuleMixin(ParametersSplitsModuleMixin, LightningModule, ABC):
    # TODO: Implement a default splitter
    def configure_optimizers(self):
        if self.opt is None:
            raise RuntimeError("You probably need to call `prepare_optimizers`")
        if self.sched is None:
            return self.opt
        return [self.opt], [self.sched]

    def prepare_optimizer(self, opt_func, lr: Union[float, slice], sched_fn=None):
        self.opt = opt_func(self.get_optimizer_param_groups(lr), self.get_lrs(lr)[0])
        self.sched = sched_fn(self.opt) if sched_fn is not None else None

    # TODO: how to be sure all parameters got assigned a lr?
    # already safe because of the check on _rcnn_pgs?
    def get_optimizer_param_groups(self, lr):
        lrs = self.get_lrs(lr)
        return [
            {"params": params, "lr": lr}
            for params, lr in zip(self.trainable_params_splits(), lrs)
        ]

    def get_lrs(self, lr):
        n_splits = len(list(self.trainable_params_splits()))
        if isinstance(lr, numbers.Number):
            return [lr] * n_splits
        elif isinstance(lr, slice):
            return np.linspace(lr.start, lr.stop, n_splits).tolist()
        else:
            raise ValueError(
                f"lr type {type(lr)} not supported, use a number or a slice"
            )
