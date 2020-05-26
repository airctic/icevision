__all__ = ["show_pred", "show_preds"]

from ..imports import *
from ..utils import *
from ..core import *
from .show_annotation import show_annotation


def show_pred(im, pred, mask_thresh=0.5, ax=None):
    # TODO: Implement keypoint
    bboxes, masks, kpts = None, None, None
    if "boxes" in pred:
        bboxes = [BBox.from_xyxy(*o) for o in pred["boxes"]]
    if "masks" in pred:
        masks = MaskArray(to_np((pred["masks"] > 0.5).long()[:, 0, :, :]))
    return show_annotation(im, bboxes=bboxes, masks=masks, ax=ax)


def show_preds(ims, preds, mask_thresh=0.5):
    return grid2(
        [
            partial(show_pred, im=im, pred=pred, mask_thresh=mask_thresh)
            for im, pred in zip(ims, preds)
        ]
    )
