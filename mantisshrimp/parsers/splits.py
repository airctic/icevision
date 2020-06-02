__all__ = ["DataSplit", "random_split", "random_split2"]

from ..imports import *


class DataSplit:
    Train = 0
    Valid = 1
    Test = 2


def random_split(p=None, splits=None):
    p = p or [0.8, 0.2]
    splits = splits or [DataSplit.Train, DataSplit.Valid, DataSplit.Test][: len(p)]
    return np.random.choice(splits, p=p)


def random_split2(ids, p):
    # calculate split indexes
    p = np.array(p) * len(ids)  # convert percentage to absolute
    p = np.ceil(p).astype(int)  # round up, so each split has at least one example
    p[p.argmax()] -= sum(p) - len(ids)  # removes excess from split with most items
    p = np.cumsum(p)

    shuffled = np.random.permutation(list(ids))
    return np.split(shuffled, p.tolist())[:-1]  # last element is always empty
