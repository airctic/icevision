import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.nn import Parameter

from typing import Collection, Union, List, Tuple
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

    def set_param_grad_state(
        self,
        stem: bool,
        bbone_blocks: Tuple[Collection[int], bool],
        neck: bool,
        bbox_head: bool,
        classifier_heads: bool,
    ):
        error_msg = f"""
        `bbone_blocks` must be a list|tuple where the second value is the gradient state to be set, and the
        first value is a List[int] between 0-9 specifying which blocks to set this state for
        """
        if not (isinstance(bbone_blocks, (list, tuple)) and len(bbone_blocks) == 2):
            raise TypeError(error_msg)
        if not isinstance(bbone_blocks[0], (list, tuple)):
            raise TypeError(error_msg)
        if not all(isinstance(x, int) for x in bbone_blocks[0]):
            raise TypeError(error_msg)
        if not 0 <= bbone_blocks[0] <= 9:
            raise ValueError(error_msg)

        for p in flatten(self._get_params_stem()):
            p.requires_grad = stem

        target_blocks, grad_state = bbone_blocks
        pgs = np.array(self._get_params_backbone())
        for p in flatten(pgs[target_blocks]):
            p.requires_grad = grad_state

        for p in flatten(self._get_params_neck()):
            p.requires_grad = neck

        for p in flatten(self._get_params_bbox_head()):
            p.requires_grad = bbox_head

        for p in flatten(self._get_params_classifier_heads()):
            p.requires_grad = classifier_heads

    def freeze(
        self,
        stem: bool = True,
        bbone_blocks: int = 0,  # between 0-9
        neck: bool = False,
        bbox_head: bool = False,
        classifier_heads: bool = False,
    ):
        """
        Freeze selected parts of the network

        Args:
            stem (bool, optional): Freeze the first conv layer. Defaults to True.
            bbone_blocks (int, optional): Number of blocks to freeze. If 0, none are frozen; if 9, all are frozen. If 3, the first 3 blocks are frozen
            neck (bool, optional): Freeze the neck (FPN). Defaults to False.
            bbox_head (bool, optional): Freeze the bounding box head (the `Detect` module). Defaults to False.
            classifier_heads (bool, optional): Freeze all the classification heads. Defaults to False.
        """
        self.set_param_grad_state(
            stem=not stem,  # If `stem==True`, set requires_grad to False
            bbone_blocks=([i for i in range(bbone_blocks)], False),
            neck=not neck,
            bbox_head=not bbox_head,
            classifier_heads=not classifier_heads,
        )

    def unfreeze(
        self,
        stem: bool = False,
        bbone_blocks: int = 9,
        neck: bool = True,
        bbox_head: bool = True,
        classifier_heads: bool = True,
    ):
        "Unfreeze specific parts of the model. By default all parts but the stem are unfrozen"
        self.set_param_grad_state(
            stem=stem,
            bbone_blocks=([i for i in range(9 - bbone_blocks, 9)], True),
            neck=neck,
            bbox_head=bbox_head,
            classifier_heads=classifier_heads,
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
