from typing import Dict
from .head import CLASSIFICATION_HEADS
import torch.nn as nn

__all__ = ["build_classifier_heads"]

# Enter dict of dicts as `cfg`
def build_classifier_heads(cfg: Dict[str, Dict[str, dict]]):
    """
    Build classification head from a config which is
    a dict of dicts. A head is created for each key in the
    input dictionary

    Returns a `nn.ModuleDict()` mapping keys from `cfg` to
    classifier heads
    """
    heads = nn.ModuleDict()
    # if cfg is not None:
    for name, config in cfg.items():
        head = CLASSIFICATION_HEADS.build(config)
        heads.update({name: head})
    return heads
