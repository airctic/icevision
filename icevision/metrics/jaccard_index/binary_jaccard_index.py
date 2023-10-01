__all__ = ["BinaryJaccardIndex"]

from icevision.imports import *
from icevision.utils import *
from icevision.data import *
from icevision.metrics.metric import *


class BinaryJaccardIndex(Metric):
    """Jaccard Index for Semantic Segmentation
    Calculates Jaccard Index for semantic segmentation (binary images only).
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._union = 0
        self._intersection = 0

    def accumulate(self, preds):

        pred = (
            np.stack([x.pred.segmentation.mask_array.data for x in preds])
            .astype(bool)
            .flatten()
        )

        target = (
            np.stack([x.ground_truth.segmentation.mask_array.data for x in preds])
            .astype(bool)
            .flatten()
        )

        self._union += np.logical_or(pred, target).sum()
        self._intersection += np.logical_and(pred, target).sum()

    def finalize(self) -> Dict[str, float]:

        if self._union == 0:
            jaccard = 0

        else:
            jaccard = self._intersection / self._union

        self._reset()
        return {"binary_jaccard_value_for_fastai": jaccard}
