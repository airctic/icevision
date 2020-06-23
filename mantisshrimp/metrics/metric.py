__all__ = ["Metric"]

from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self):
        self._model = None

    @abstractmethod
    def reset(self):
        """ Resets metric internal state
        """

    @abstractmethod
    def accumulate(self, xb, yb, preds):
        """ Accumulate stats for a single batch
        """

    @abstractmethod
    def finalize(self) -> float:
        """ Called at the end of the validation loop
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__
