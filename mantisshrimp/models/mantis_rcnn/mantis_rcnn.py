__all__ = ["MantisRCNN"]

from ...imports import *
from ...utils import *
from ...core import *
from ..mantis_module import *


class MantisRCNN(MantisModule, ABC):
    def __init__(self, metrics=None):
        super().__init__()
        self.metrics = metrics or []

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
            o = metric.step(self, xb, yb, preds)
            if notnone(o):
                raise NotImplementedError  # How to update res?
        res.update({"valid/loss": loss, **losses})
        return res

    def validation_epoch_end(self, outs):
        res = {}
        for metric in self.metrics:
            o = metric.end(self, outs)
            if notnone(o):
                raise NotImplementedError  # How to update res?
        log = {k: torch.stack(v).mean() for k, v in mergeds(outs).items()}
        res.update({"val_loss": log["valid/loss"], "log": log})
        return res

    def dataloader(self, dataset, **kwargs) -> DataLoader:
        def collate_fn(data):
            ts = [self.build_training_sample(**o) for o in data]
            return list(zip(*ts))

        return DataLoader(dataset=dataset, collate_fn=collate_fn, **kwargs)

    def build_training_sample(self, *args, **kwargs):
        raise NotImplementedError
