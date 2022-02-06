__all__ = ["show_batch"]

from icevision.utils import *
from icevision.visualize import *
from icevision.models.mmdet.common.utils import *
import numpy as np
from icevision.core.mask import MaskArray


def show_batch(batch_and_records, ncols: int = 1, figsize=None, **show_samples_kwargs):
    """Show a single batch from a dataloader.
    # Arguments
        show_samples_kwargs: Check the parameters from `show_samples`
    """
    batch, records = batch_and_records

    # HACK: Show mask correctly
    # Given that records are unloaded following creation of a batch,
    # the only way to retrieve mask data is from the batch itself

    for tensor_image, record, gt_mask in zip(
        batch["img"][:], records, batch["gt_masks"]
    ):
        mask = np.stack(gt_mask.masks)
        image = mmdet_tensor_to_image(tensor_image)
        record.set_img(image)
        record.detection.set_mask_array(MaskArray(mask))

    return show_samples(records, ncols=ncols, figsize=figsize, **show_samples_kwargs)
