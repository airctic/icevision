__all__ = ["BinaryDiceCoefficient"]

from icevision.imports import *
from icevision.utils import *
from icevision.data import *
from icevision.metrics.metric import *


class BinaryDiceCoefficient(Metric):
    """Binary Dice Coefficient for Semantic Segmentation

    Calculates Dice Coefficient for semantic segmentation (binary images only).

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

        self._union += pred.sum() + target.sum()
        self._intersection += np.logical_and(pred, target).sum()

    def finalize(self) -> Dict[str, float]:

        if self._union == 0:
            dice = 0

        else:
            dice = 2.0 * self._intersection / self._union

        self._reset()
        return {"dummy_value_for_fastai": dice}
