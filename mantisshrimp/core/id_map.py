__all__ = ["IDMap"]

from mantisshrimp.imports import *


class IDMap:
    """
    Works like a dictionary that automatically assign values for new keys.
    """

    def __init__(self, imageids=None):
        self.i2imageid = imageids or []
        self.imageid2i = OrderedDict(enumerate(self.i2imageid))

    def __getitem__(self, imageid):
        try:
            i = self.imageid2i[imageid]
        except KeyError:
            i = len(self.i2imageid)
            self.imageid2i[imageid] = i
            self.i2imageid.append(imageid)
        return i
