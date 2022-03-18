__all__ = ["MulticlassDiceCoefficient"]

from icevision.imports import *
from icevision.utils import *
from icevision.data import *
from icevision.metrics.metric import *


class MulticlassDiceCoefficient(Metric):
    """Multi-class Dice Coefficient for Semantic Segmentation

    Calculates Dice Coefficient for semantic segmentation (multi-class tasks).

    Heaviliy inspired by fastai's implementation:  Implement multi-class version following https://github.com/fastai/fastai/blob/594e1cc20068b0d99bfc30bfe6dac88ab381a157/fastai/metrics.py#L343

    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._union = {}
        self._intersection = {}

    def accumulate(self, preds):

        pred = np.stack([x.pred.segmentation.mask_array.data for x in preds]).flatten()

        target = self._seg_masks_gt = np.stack(
            [x.ground_truth.segmentation.mask_array.data for x in preds]
        ).flatten()

        # We iterate through all unique classes across preditions and targets
        for c in np.unique(np.concatenate((np.unique(pred), np.unique(target)))):

            p = np.where(pred == c, 1, 0)
            t = np.where(target == c, 1, 0)

            c_inter = np.logical_and(p, t).sum()
            c_union = p.sum() + t.sum()

            if c in self._intersection:
                self._intersection[c] += c_inter
                self._union[c] += c_union
            else:
                self._intersection[c] = c_inter
                self._union[c] = c_union

    def finalize(self) -> Dict[str, float]:

        class_dice_scores = np.array([])

        for c in self._intersection:
            class_dice_scores = np.append(
                class_dice_scores,
                2.0 * self._intersection[c] / self._union[c]
                if self._union[c] > 0
                else np.nan,
            )

        dice = np.nanmean(class_dice_scores)

        self._reset()
        return {"dummy_value_for_fastai": dice}
