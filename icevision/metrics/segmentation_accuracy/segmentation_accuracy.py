__all__ = ["SegmentationAccuracy"]

from icevision.imports import *
from icevision.utils import *
from icevision.data import *
from icevision.metrics.metric import *


class SegmentationAccuracy(Metric):
    """A simple segmentation accuracy metric

    Calculates segmentation accuracy.

    # Arguments
        ignore_class: class to ignore during metric computation (e.g. void class).

    """

    _seg_masks_gt = None
    _seg_masks_pred = None

    def __init__(self, ignore_class=None):

        self.ignore_class = ignore_class

    def _reset(self):
        self._seg_masks_gt = None
        self._seg_masks_pred = None

    def accumulate(self, preds):
        if self._seg_masks_pred is None:
            self._seg_masks_pred = np.stack(
                [x.pred.segmentation.mask_array.data for x in preds]
            )
        else:
            self._seg_masks_pred = np.concatenate(
                [
                    self._seg_masks_pred,
                    np.stack([x.pred.segmentation.mask_array.data for x in preds]),
                ]
            )

        if self._seg_masks_gt is None:
            self._seg_masks_gt = np.stack(
                [x.ground_truth.segmentation.mask_array.data for x in preds]
            )
        else:
            self._seg_masks_gt = np.concatenate(
                [
                    self._seg_masks_gt,
                    np.stack(
                        [x.ground_truth.segmentation.mask_array.data for x in preds]
                    ),
                ],
            )

    def finalize(self) -> Dict[str, float]:

        if self.ignore_class is not None:
            keep_idxs = self._seg_masks_gt != self.ignore_class
            target = self._seg_masks_gt[keep_idxs]
            pred = self._seg_masks_pred[keep_idxs]
        else:
            target = self._seg_masks_gt
            pred = self._seg_masks_pred

        accuracy = (target == pred).mean()

        self._reset()
        return {"dummy_value_for_fastai": accuracy}
