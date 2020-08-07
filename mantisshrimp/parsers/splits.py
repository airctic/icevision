__all__ = [
    "DataSplitter",
    "SingleSplitSplitter",
    "RandomSplitter",
]

from mantisshrimp.imports import *
from mantisshrimp.utils import *


class DataSplitter(ABC):
    """Base class for all data splitters.
    """

    def __call__(self, ids):
        return self.split(ids)

    @abstractmethod
    def split(self, ids):
        """Splits `ids` into groups.

        # Arguments
            ids: Items to be splitted.
        """
        pass


class SingleSplitSplitter(DataSplitter):
    """Return all items in a single group, without shuffling.
    """

    def split(self, ids):
        """Puts all `ids` in a single group.

        # Arguments
            ids: Items to be splitted.
        """
        return [ids]


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

    def split(self, ids):
        """Randomly splits `ids` based on parameters passed to the constructor of this class.

        # Arguments
            ids: Items to be splitted.
        """
        # calculate split indexes
        p = np.array(self.probs) * len(ids)  # convert percentage to absolute
        p = np.ceil(p).astype(int)  # round up, so each split has at least one example
        p[p.argmax()] -= sum(p) - len(ids)  # removes excess from split with most items
        p = np.cumsum(p)

        with np_local_seed(self.seed):
            shuffled = np.random.permutation(list(ids))
        return np.split(shuffled, p.tolist())[:-1]  # last element is always empty
