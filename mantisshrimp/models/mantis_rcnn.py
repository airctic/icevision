__all__ = ["MantisRCNN"]

from ..imports import *
from ..utils import *
from .utils import *

# TODO: Refactor to use mixins
class MantisRCNN(LightningModule):
    def __init__(self, model_splitter=None, metrics=None):
        super().__init__()
        self.metrics = metrics or []
        for metric in self.metrics:
            metric.register_model(self)
        self._create_model()
        self.model_splits = self._get_model_splits()

    def _create_model(self):
        raise NotImplementedError

    def _get_model_splits(self):
        raise NotImplementedError

    def predict(self, ims=None, rs=None):
        if bool(ims) == bool(rs):
            raise ValueError("You should either pass ims or rs")
        if notnone(rs):
            ims = [open_img(o.info.filepath) for o in rs]
        xs = [im2tensor(o).to(model_device(self)) for o in ims]
        self.eval()
        return ims, self(xs)

    def training_step(self, b, b_idx):
        xb, yb = b
        losses = self(xb, list(yb))
        loss = sum(losses.values())
        log = {"train/loss": loss, **{f"train/{k}": v for k, v in losses.items()}}
        return {"loss": loss, "log": log}

    def validation_step(self, b, b_idx):
        xb, yb = b
        with torch.no_grad():
            self.train()
            losses = self(xb, list(yb))
            self.eval()
            preds = self(xb)
        loss = sum(losses.values())
        losses = {f"valid/{k}": v for k, v in losses.items()}
        res = {}
        for metric in self.metrics:
            o = metric.step(xb, yb, preds)
            if notnone(o):
                raise NotImplementedError  # How to update res?
        res.update({"valid/loss": loss, **losses})
        return res

    def validation_epoch_end(self, outs):
        res = {}
        for metric in self.metrics:
            o = metric.end(outs)
            if notnone(o):
                raise NotImplementedError  # How to update res?
        log = {k: torch.stack(v).mean() for k, v in mergeds(outs).items()}
        res.update({"val_loss": log["valid/loss"], "log": log})
        return res

    # TODO: Add error "First need to call prepare optimizer"
    def configure_optimizers(self):
        if self.sched is None:
            return self.opt
        return [self.opt], [self.sched]

    def prepare_optimizers(self, opt_func, lr, sched_fn=None):
        self.opt = opt_func(self.get_optimizer_param_groups(lr), self.get_lrs(lr)[0])
        self.sched = sched_fn(self.opt) if sched_fn is not None else None

    # TODO: how to be sure all parameters got assigned a lr?
    # already safe because of the check on _rcnn_pgs?
    def get_optimizer_param_groups(self, lr):
        lrs = self.get_lrs(lr)
        return [{"params": params, "lr": lr} for params, lr in zip(self.trainable_params_splits(), lrs)]

    def get_lrs(self, lr):
        n_splits = len(list(self.trainable_params_splits()))
        if isinstance(lr, numbers.Number):
            return [lr] * n_splits
        elif isinstance(lr, slice):
            return np.linspace(lr.start, lr.stop, n_splits).tolist()
        else:
            raise ValueError(f"lr type {type(lr)} not supported, use a number or a slice")

    # def freeze(self, n:int=None):

    def trainable_params_splits(self):
        """ Get trainable parameters from parameter groups
            If a parameter group does not have trainable params, it does not get added
        """
        for split in self.model_splits:
            params = list(filter_params(split, only_trainable=True))
            if params:
                yield params
