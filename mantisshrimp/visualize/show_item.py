from ..utils import *
from .show_annotation import *

def show_item(o, *, label=True, bbox=True, mask=True, catmap=None, ax=None):
    names = [catmap.i2o[i].name for i in o.labels] if notnone(catmap) else o.labels
    show_annotation(im=o.img, ax=ax,
               labels=names if label else None,
               bboxes=o.bboxes if bbox else None,
               masks=o.masks if mask else None)
