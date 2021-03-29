__all__ = ["TorchvisionBackboneConfig"]

from icevision.imports import *
from icevision.backbones import BackboneConfig


class TorchvisionBackboneConfig(BackboneConfig):
    def __init__(self, backbone_fn, **backbone_fn_kwargs):
        self.backbone: nn.Module

        self.backbone_fn = backbone_fn
        self.backbone_fn_kwargs = backbone_fn_kwargs

    def __call__(self, pretrained: bool = True, **kwargs):
        self.pretrained = pretrained

        # kwargs passed to call overwrite kwargs from init
        kwargs = {**self.backbone_fn_kwargs, **kwargs}
        self.backbone = self.backbone_fn(pretrained=pretrained, **kwargs)

        return self
