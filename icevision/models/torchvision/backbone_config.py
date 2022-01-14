__all__ = ["TorchvisionBackboneConfig"]

from icevision.imports import *
from icevision.backbones import BackboneConfig


class TorchvisionBackboneConfig(BackboneConfig):
    def __init__(self, backbone_fn, **backbone_fn_kwargs):
        self.backbone: nn.Module

        self.backbone_fn = backbone_fn
        self.backbone_fn_kwargs = backbone_fn_kwargs

    def __call__(self, pretrained: bool = True, **kwargs):
        """Completes the configuration of the backbone

        # Arguments
            pretrained: If True, use a pretrained backbone (on COCO).
            By default it is set to True: this is generally used when training a new model (transfer learning).
            `pretrained = False`  is used during inference (prediction) for cases where the users have their own pretrained weights.
        """
        self.pretrained = pretrained

        # kwargs passed to call overwrite kwargs from init
        kwargs = {**self.backbone_fn_kwargs, **kwargs}
        self.backbone = self.backbone_fn(pretrained=pretrained, **kwargs)

        return self
