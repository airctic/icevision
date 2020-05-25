__all__ = ['MantisRCNN']

from ..imports import *
from ..utils import *

# TODO: Refactor to use mixins
class MantisRCNN(LightningModule):
    def __init__(self, model_splitter=None, metrics=None):
        super().__init__()
        self.metrics = metrics or []
        for metric in self.metrics: metric.register_model(self)
        self._create_model()
        self.pgs = self._get_pgs()

    def predict(self, ims=None, rs=None):
        if bool(ims) == bool(rs): raise ValueError('You should either pass ims or rs')
        if notnone(rs): ims = [open_img(o.info.filepath) for o in rs]
        xs = [im2tensor(o).to(model_device(self)) for o in ims]
        self.eval()
        return ims, self(xs)

    def training_step(self, b, b_idx):
        xb, yb = b
        losses = self(xb, list(yb))
        loss = sum(losses.values())
        log = {'train/loss': loss, **{f'train/{k}': v for k, v in losses.items()}}
        return {'loss': loss, 'log': log}

    def validation_step(self, b, b_idx):
        xb, yb = b
        with torch.no_grad():
            self.train();
            losses = self(xb, list(yb))
            self.eval();
            preds = self(xb)
        loss = sum(losses.values())
        losses = {f'valid/{k}': v for k, v in losses.items()}
        res = {}
        for metric in self.metrics:
            o = metric.step(xb, yb, preds)
            if notnone(o): raise NotImplementedError  # How to update res?
        res.update({'valid/loss': loss, **losses})
        return res

    def validation_epoch_end(self, outs):
        res = {}
        for metric in self.metrics:
            o = metric.end(outs)
            if notnone(o): raise NotImplementedError  # How to update res?
        log = {k: torch.stack(v).mean() for k, v in mergeds(outs).items()}
        res.update({'val_loss': log['valid/loss'], 'log': log})
        return res

    # TODO: Add error "First need to call prepare optimizer"
    def configure_optimizers(self):
        if self.sched is None: return self.opt
        return [self.opt], [self.sched]

    def prepare_optimizers(self, opt_func, lr, sched_fn=None):
        self.opt = opt_func(self.get_ps(lr), self.get_lrs(lr)[0])
        self.sched = sched_fn(self.opt) if sched_fn is not None else None

    # TODO: how to be sure all parameters got assigned a lr?
    # already safe because of the check on _rcnn_pgs?
    def get_ps(self, lr):
        lrs = self.get_lrs(lr)
        return [{'params': params(pg), 'lr': lr} for pg, lr in zipsafe(self.pgs, lrs)]

    def get_lrs(self, lr):
        if isinstance(lr, numbers.Number):
            return [lr] * len(self.pgs)
        elif isinstance(lr, slice):
            return np.linspace(lr.start, lr.stop, len(self.pgs)).tolist()
        else:
            raise ValueError(f'lr type {type(lr)} not supported, use a number or a slice')
