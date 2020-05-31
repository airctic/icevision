__all__ = ["Transform"]

from ..imports import *
from ..core import *


class Transform(ABC):
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, item: Item):
        tfmed = self.apply(**item.asdict())
        return item.replace(**tfmed)

    @abstractmethod
    def apply(
        self,
        img: np.ndarray,
        imageid: int,
        labels: List[int],
        iscrowds: List[int],
        bboxes: List[BBox],
        masks: MaskArray,
    ):
        """ Apply the transform
        Returns:
              dict: Modified values, the keys of the dictionary should have the same
              names as the keys received by this function
        """
