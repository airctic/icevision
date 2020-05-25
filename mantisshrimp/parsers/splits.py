__all__ = ['DataSplit', 'random_split']

from ..imports import *

class DataSplit:
    Train = 0
    Valid = 1
    Test = 2

def random_split(p=None, splits=None):
    p = p or [.8, .2]
    splits = splits or [DataSplit.Train, DataSplit.Valid, DataSplit.Test][:len(p)]
    return np.random.choice(splits, p=p)
