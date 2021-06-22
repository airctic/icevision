import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter

from typing import Union, List
from icevision.utils.torch_utils import params
from icevision.utils.utils import flatten
from loguru import logger

logger = logger.opt(colors=True)

__all__ = ["FreezingInterfaceExtension"]


class FreezingInterfaceExtension:
    """
    Model freezing and unfreezing extensions for `HybridYOLOV5`
    Note that the BatchNorm layers are also frozen, but that part is not
    defined here, but in the main module's `.train()` method directly
    """

    def _get_params_stem(self) -> List[nn.Parameter]:
        return params(self.model[0])

    def _get_params_backbone(self) -> List[List[Parameter]]:
        return [params(m) for m in self.model[1:10]]

    def _get_params_neck(self) -> List[List[Parameter]]:
        return [params(m) for m in self.model[10:][:-1]]

    def _get_params_bbox_head(self) -> List[List[Parameter]]:
        return params(self.model[-1])

    def _get_params_classifier_heads(self) -> List[List[Parameter]]:
        return [params(self.classifier_heads)]

    def freeze(
        self,
        freeze_stem: bool = True,
        freeze_bbone_blocks_until: int = 0,  # between 0-9
        freeze_neck: bool = False,
        freeze_bbox_head: bool = False,
        freeze_classifier_heads: bool = False,
        _grad: bool = False,  # Don't modify.
    ):
        """
        Freeze selected parts of the network

        Args:
            freeze_stem (bool, optional): Freeze the first conv layer. Defaults to True.
            freeze_bbone_blocks_until (int, optional): Number of blocks to freeze. If 0, none are frozen; if 9, all are frozen. Defaults to 0.
            freeze_neck (bool, optional): Freeze the neck (FPN). Defaults to False.
            freeze_bbox_head (bool, optional): Freeze the bounding box head (the `Detect` module). Defaults to False.
            freeze_classifier_heads (bool, optional): Freeze all the classification heads. Defaults to False.
            _grad (bool): DO NOT MODIFY this argument. It is used internally for `.unfreeze()`
        """
        if freeze_stem:
            for p in flatten(self._get_params_stem()):
                p.requires_grad = _grad

        assert 0 <= freeze_bbone_blocks_until <= 9, "Num blocks must be between 0-9"
        for i, pg in enumerate(self._get_params_backbone(), start=1):
            if i > freeze_bbone_blocks_until:
                break
            else:
                for p in pg:
                    p.requires_grad = _grad

        if freeze_neck:
            for p in flatten(self._get_params_neck()):
                p.requires_grad = _grad

        if freeze_bbox_head:
            for p in flatten(self._get_params_bbox_head()):
                p.requires_grad = _grad

        if freeze_classifier_heads:
            for p in flatten(self._get_params_classifier_heads()):
                p.requires_grad = _grad

    def unfreeze(
        self,
        unfreeze_stem: bool = False,
        unfreeze_bbone_blocks_until: int = 9,  # either 0-9 TODO FIXME
        unfreeze_neck: bool = True,
        unfreeze_bbox_head: bool = True,
        unfreeze_classifier_heads: bool = True,
    ):
        "Unfreeze specific parts of the model. By default all parts but the stem are unfrozen"
        if not unfreeze_bbone_blocks_until in [0, 9]:
            raise RuntimeError(
                f"Currently we can only unfreeze all or no blocks at once. Pass `unfreeze_bbone_blocks_until=9 | 0` to do so"
            )

        self.freeze(
            freeze_stem=unfreeze_stem,
            freeze_bbone_blocks_until=unfreeze_bbone_blocks_until,
            freeze_neck=unfreeze_neck,
            freeze_bbox_head=unfreeze_bbox_head,
            freeze_classifier_heads=unfreeze_classifier_heads,
            _grad=True,
        )

    def freeze_specific_classifier_heads(
        self, names: Union[str, List[str], None] = None, _grad: bool = False
    ):
        "Freeze all, one or a few classifier heads"
        if isinstance(names, str):
            names = [names]
        if names is None:
            names = list(self.classifier_heads.keys())

        for name in names:
            for p in flatten(params(self.classifier_heads[name])):
                p.requires_grad = _grad

    def unfreeze_specific_classifier_heads(
        self, names: Union[str, List[str], None] = None
    ):
        self.freeze_specific_classifier_heads(names=names, _grad=True)
