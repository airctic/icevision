__all__ = ["BackboneConfig"]

from icevision.imports import *


class BackboneConfig(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> "BackboneConfig":
        """Completes configuration for this backbone.

        Called by the end user. All heavy lifting (such as creating a NN backbone)
        should be done here.
        """
