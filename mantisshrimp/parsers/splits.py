__all__ = ["DataSplit", "random_split", "RandomSplitter"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *


class DataSplit:
    Train = 0
    Valid = 1
    Test = 2


def random_split(p=None, splits=None):
    p = p or [0.8, 0.2]
    splits = splits or [DataSplit.Train, DataSplit.Valid, DataSplit.Test][: len(p)]
    return np.random.choice(splits, p=p)


def RandomSplitter(probs, seed=None):
    def _inner(ids):
        # calculate split indexes
        p = np.array(probs) * len(ids)  # convert percentage to absolute
        p = np.ceil(p).astype(int)  # round up, so each split has at least one example
        p[p.argmax()] -= sum(p) - len(ids)  # removes excess from split with most items
        p = np.cumsum(p)

        with np_local_seed(seed):
            shuffled = np.random.permutation(list(ids))
        return np.split(shuffled, p.tolist())[:-1]  # last element is always empty

    return _inner
