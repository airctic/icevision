__all__ = ['rcnn_collate', 'RCNNDataLoader']

from ..imports import *
from ..utils import *
from ..core import *

def rcnn_collate(items):
    ts = [item2tensor(o) for o in items]
    return list(zip(*ts))

RCNNDataLoader = partial(DataLoader, collate_fn=rcnn_collate)
