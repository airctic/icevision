__all__ = ['InfoParser']

from ..imports import *
from ..utils import *
from ..core import *
from .splits import *


class InfoParser:
    def __init__(self, data, source=None, idmap=None):
        self.data, self.source = data, Path(source or '.')
        self.idmap = idmap or IDMap()

    def __iter__(self): yield from self.data
    def __len__(self):  return len(self.data)

    def prepare(self, o): pass
    def imageid(self, o): raise NotImplementedError
    def filepath(self, o): raise NotImplementedError
    def h(self, o): raise NotImplementedError
    def w(self, o): raise NotImplementedError
    def split(self, o): return random_split()

    def parse(self, show_pbar=True):
        xs,imageids = [],set()
        for o in pbar(self, show_pbar):
            self.prepare(o)
            imageid = self.idmap[self.imageid(o)]
            if imageid not in imageids:
                imageids.add(imageid)
                xs.append(ImageInfo(imageid=imageid, filepath=self.filepath(o),
                                    split=self.split(o), h=self.h(o), w=self.w(o)))
        return xs
