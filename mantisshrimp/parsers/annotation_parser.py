__all__ = ['AnnotationParser']

from ..imports import *
from ..utils import *
from ..core import *

class AnnotationParser:
    def __init__(self, data, source, catmap, idmap=None):
        self.data, self.source, self.catmap = data, source, catmap
        self.idmap = idmap or IDMap()

    def __iter__(self): yield from self.data
    def __len__(self): return len(self.data)

    # Methods to override
    def prepare(self, o): pass
    def imageid(self, o): raise NotImplementedError
    def label(self, o): raise NotImplementedError
    def bbox(self, o): pass
    def mask(self, o): pass
    def iscrowd(self, o): return 0

    def parse(self, show_pbar=True):
        imageids = set()
        labels = defaultdict(list)
        iscrowds = defaultdict(list)
        bboxes = defaultdict(list)
        masks = defaultdict(list)
        for o in pbar(self, show_pbar):
            self.prepare(o) # TODO: Refactor with python 3.8 walrus syntax
            imageid = self.idmap[self.imageid(o)]
            labels[imageid].extend([self.catmap.id2i[id] for id in L(self.label(o))])
            iscrowds[imageid].extend(L(self.iscrowd(o)))
            bbox=self.bbox(o)
            if notnone(bbox): bboxes[imageid].extend(L(bbox))
            mask = self.mask(o)
            if notnone(mask): masks[imageid].extend(L(mask))
            imageids.add(imageid)
        for d in [bboxes, masks, labels, iscrowds]: d.default_factory = lambda: None
        return [Annotation(i, labels[i], bboxes=bboxes[i], masks=masks[i], iscrowds=iscrowds[i]) for i in imageids]

