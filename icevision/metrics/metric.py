__all__ = ["Metric"]

from icevision.imports import *


class Metric(ABC):
    def __init__(self):
        self._model = None

    @abstractmethod
    def accumulate(self, records, preds):
        """Accumulate stats for a single batch"""

    @abstractmethod
    def finalize(self) -> Dict[str, float]:
        """Called at the end of the validation loop"""

    @property
    def name(self) -> str:
        return self.__class__.__name__
