__all__ = [
    "DataSplitter",
    "SingleSplitSplitter",
    "RandomSplitter",
    "FixedSplitter",
]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *


class DataSplitter(ABC):
    """Base class for all data splitters.
    """

    def __call__(self, idmap: IDMap):
        return self.split(idmap=idmap)

    @abstractmethod
    def split(self, idmap: IDMap):
        """Splits `ids` into groups.

        # Arguments
            idmap: idmap used for getting ids.
        """
        pass


class SingleSplitSplitter(DataSplitter):
    """Return all items in a single group, without shuffling.
    """

    def split(self, idmap: IDMap):
        """Puts all `ids` in a single group.

        # Arguments
            idmap: idmap used for getting ids.
        """
        return [idmap.get_ids()]


class RandomSplitter(DataSplitter):
    """Randomly splits items.

    # Arguments
        probs: `Sequence` of probabilities that must sum to one. The length of the 
            `Sequence` is the number of groups to to split the items into.
        seed: Internal seed used for shuffling the items. Define this if you need
            reproducible results.
    """

    def __init__(self, probs: Sequence[int], seed: int = None):
        self.probs = probs
        self.seed = seed

    def split(self, idmap: IDMap):
        """Randomly splits `ids` based on parameters passed to the constructor of this class.

        # Arguments
            idmap: idmap used for getting ids.
        """
        ids = idmap.get_ids()
        # calculate split indexes
        p = np.array(self.probs) * len(ids)  # convert percentage to absolute
        p = np.ceil(p).astype(int)  # round up, so each split has at least one example
        p[p.argmax()] -= sum(p) - len(ids)  # removes excess from split with most items
        p = np.cumsum(p)

        with np_local_seed(self.seed):
            shuffled = np.random.permutation(list(ids))
        return np.split(shuffled, p.tolist())[:-1]  # last element is always empty


class FixedSplitter(DataSplitter):
    """ Split `ids` based on predefined splits.

    # Arguments:
        splits: The predefined splits.
    """

    def __init__(self, splits: Sequence[Sequence[Hashable]]):
        self.splits = splits

    def split(self, idmap: IDMap):
        """Execute the split

        # Arguments
            idmap: idmap used for getting ids.
        """
        return [[idmap.get_name(name) for name in names] for names in self.splits]
