__all__ = [
    "EfficientNet_B1",
    "EfficientNet_B2",
    "EfficientNet_B3",
    "EfficientNet_B4",
    "EfficientNet_Lite0",
]

from icevision.models.mmdet.backbones.timm.common import *
from mmdet.models.builder import BACKBONES

from typing import Optional, Collection
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Tuple, Collection, List

import timm
from timm.models.mobilenetv3 import *
from timm.models.registry import *

import torch.nn as nn
import torch


class BaseEfficientNet(MMDetTimmBase):
    """
    Base class that implements model freezing and forward methods
    """

    def __init__(
        self,
        model_name: str = None,
        pretrained: bool = True,
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):

        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
        )
        self.frozen_stages = frozen_stages
        self.frozen_stem = frozen_stem

    def test_pretrained_weights(self):
        # Get model method from the timm registry by model name
        model_fn = model_entrypoint(self.model_name)
        model = model_fn(pretrained=self.pretrained)
        assert torch.equal(self.model.conv_stem.weight, model.conv_stem.weight)

    def post_init_setup(self):
        self.freeze(
            freeze_stem=self.frozen_stem,
            freeze_blocks=self.frozen_stages,
        )
        # self.test_pretrained_weights()

    def freeze(self, freeze_stem: bool = True, freeze_blocks: int = 1):
        "Optionally freeze the stem and/or Inverted Residual blocks of the model"
        if 0 > freeze_blocks > 8:
            raise ValueError("freeze_blocks values must between 0 and 7 included")

        m = self.model

        # Stem freezing logic
        if freeze_stem:
            for l in [m.conv_stem, m.bn1]:
                l.eval()
                for param in l.parameters():
                    param.requires_grad = False

        # `freeze_blocks=1` freezes the first block, and so on
        for i, block in enumerate(m.blocks, start=1):
            if i > freeze_blocks:
                break
            else:
                block.eval()
                for param in block.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        "Convert the model to training mode while optionally freezing BatchNorm"
        super(BaseEfficientNet, self).train(mode)
        self.freeze(
            freeze_stem=self.frozen_stem,
            freeze_blocks=self.frozen_stages,
        )
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module(force=True)
class EfficientNet_B0(BaseEfficientNet):
    def __init__(
        self,
        pretrained: bool = True,
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "EfficientNet B0"
        super().__init__(
            model_name="efficientnet_b0",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )

        self.weights_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth"

        self.post_init_setup()


@BACKBONES.register_module(force=True)
class EfficientNet_B1(BaseEfficientNet):
    def __init__(
        self,
        pretrained: bool = True,
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "EfficientNet B1"
        super().__init__(
            model_name="efficientnet_b1",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )
        self.weights_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth"

        self.post_init_setup()


@BACKBONES.register_module(force=True)
class EfficientNet_B2(BaseEfficientNet):
    def __init__(
        self,
        pretrained: bool = True,
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "EfficientNet B2"
        super().__init__(
            model_name="efficientnet_b2",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )
        self.weights_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth"

        self.post_init_setup()


@BACKBONES.register_module(force=True)
class EfficientNet_B3(BaseEfficientNet):
    def __init__(
        self,
        pretrained: bool = True,
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "EfficientNet B3"
        super().__init__(
            model_name="efficientnet_b3",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )
        self.weights_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth"

        self.post_init_setup()


@BACKBONES.register_module(force=True)
class EfficientNet_B4(BaseEfficientNet):
    def __init__(
        self,
        pretrained: bool = True,
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "EfficientNet B4"
        super().__init__(
            model_name="efficientnet_b2",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )
        self.weights_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth"

        self.post_init_setup()


@BACKBONES.register_module(force=True)
class EfficientNet_Lite0(BaseEfficientNet):
    def __init__(
        self,
        pretrained: bool = True,
        out_indices: Collection[int] = (2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "EfficientNet Lite0"
        super().__init__(
            model_name="efficientnet_lite0",
            pretrained=pretrained,
            out_indices=out_indices,
            norm_eval=norm_eval,
            frozen_stages=frozen_stages,
            frozen_stem=frozen_stem,
        )
        self.weights_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth"

        self.post_init_setup()
