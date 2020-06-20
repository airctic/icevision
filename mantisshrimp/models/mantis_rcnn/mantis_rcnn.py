__all__ = ["MantisRCNN"]

from mantisshrimp.imports import *
from ...utils import *
from mantisshrimp.core import *
from ..mantis_module import *
from mantisshrimp.backbones import *


class MantisRCNN(MantisModule, ABC):
    def __init__(self, metrics=None):
        super().__init__()
        self.metrics = metrics or []

    @staticmethod
    @abstractmethod
    def build_training_sample(*args, **kwargs):
        """
        Converts a record to a format understood by the model.
        """

    @staticmethod
    def loss(preds, targs) -> Tensor:
        return sum(preds.values())

    def get_logs(self, batch, preds) -> dict:
        # losses are the logs
        return preds

    def predict(self, ims=None, rs=None):
        if bool(ims) == bool(rs):
            raise ValueError("You should either pass ims or rs")
        if notnone(rs):
            ims = [open_img(o.info.filepath) for o in rs]
        xs = [im2tensor(o).to(model_device(self)) for o in ims]
        self.eval()
        return ims, self(xs)

    @classmethod
    def collate_fn(cls, data):
        ts = [cls.build_training_sample(**o) for o in data]
        xb, yb = zip(*ts)
        return xb, list(yb)

    @classmethod
    def dataloader(cls, dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset=dataset, collate_fn=cls.collate_fn, **kwargs)

    @staticmethod
    def get_backbone_by_name(
        name: str, fpn: bool = True, pretrained: bool = True
    ) -> nn.Module:
        """
        Args:
            backbone (str): If none creates a default resnet50_fpn model trained on MS COCO 2017
                Supported backones are: "resnet18", "resnet34","resnet50", "resnet101", "resnet152",
                 "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2", as resnets with fpn backbones.
                Without fpn backbones supported are: "resnet18", "resnet34", "resnet50","resnet101",
                 "resnet152", "resnext101_32x8d", "mobilenet", "vgg11", "vgg13", "vgg16", "vgg19",
            pretrained (bool): Creates a pretrained backbone with imagenet weights.
        """
        # Giving string as a backbone, which is either supported resnet or backbone
        if fpn:
            # Creates a torchvision resnet model with fpn added
            # It returns BackboneWithFPN model
            backbone = resnet_fpn_backbone(name, pretrained=pretrained)
        else:
            # This does not create fpn backbone, it is supported for all models
            backbone = create_torchvision_backbone(name, pretrained=pretrained)
        return backbone


class LightningRCNN:
    def training_step(self, batch, batch_idx):
        xb, yb = batch
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
            o = metric.accumulate(self, xb, yb, preds)
            if notnone(o):
                raise NotImplementedError  # How to update res?
        res.update({"valid/loss": loss, **losses})
        return res

    def validation_epoch_end(self, outs):
        res = {}
        for metric in self.metrics:
            o = metric.finalize(self, outs)
            if notnone(o):
                raise NotImplementedError  # How to update res?
        log = {k: torch.stack(v).mean() for k, v in mergeds(outs).items()}
        res.update({"val_loss": log["valid/loss"], "log": log})
        return res
