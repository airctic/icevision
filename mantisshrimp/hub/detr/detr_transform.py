from mantisshrimp.imports import *
from mantisshrimp import *


class DetrTransform(Transform):
    def __init__(self, tfm):
        self.tfm = tfm

    def apply(
        self,
        img: np.ndarray,
        imageid: int,
        label: List[int],
        iscrowd: List[int],
        bbox: List[BBox],
        mask: MaskArray,
    ):
        pass
