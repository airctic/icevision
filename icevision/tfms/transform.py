__all__ = ["Transform"]

from icevision.imports import *
from icevision.core import *


class Transform(ABC):
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, data: dict):
        data = data.copy()
        tfmed = self.apply(**data)
        data.update(tfmed)
        return data

    @abstractmethod
    def apply(
        self,
        img: np.ndarray,
        imageid: int,
        label: List[int],
        iscrowd: List[int],
        bbox: List[BBox],
        mask: MaskArray,
    ):
        """Apply the transform
        Returns:
              dict: Modified values, the keys of the dictionary should have the same
              names as the keys received by this function
        """
