__all__ = ["IDMap"]

from ..imports import *


class IDMap:
    def __init__(self, imageids=None):
        self.i2imageid = imageids or []
        self.imageid2i = OrderedDict(enumerate(self.i2imageid))

    def __getitem__(self, imageid):
        try:
            return self.imageid2i[imageid]
        except KeyError:
            self.imageid2i[imageid] = len(self.i2imageid)
            self.i2imageid.append(imageid)
        return self[imageid]
