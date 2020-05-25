__all__ = ['rcnn_param_groups']

from ..imports import *
from ..utils import *

def rcnn_param_groups(m):
    body = m.backbone.body
    pgs  = []
    pgs += [nn.Sequential(body.conv1, body.bn1)]
    pgs += [getattr(body,l) for l in list(body) if l.startswith('layer')]
    pgs += [m.backbone.fpn, m.rpn, m.roi_heads]
    if len(params(nn.Sequential(*pgs))) != len(params(m)):
        raise RuntimeError('Malformed model parameters groups, you probably need to use a custom model_splitter')
    return pgs
