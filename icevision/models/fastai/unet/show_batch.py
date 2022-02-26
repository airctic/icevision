__all__ = ["show_batch"]

from icevision.utils import *
from icevision.visualize import *
from icevision.models.utils import *
from icevision.core import *


def show_batch(batch_and_records, ncols: int = 1, figsize=None, **show_samples_kwargs):
    """Show a single batch from a dataloader.

    # Arguments
        show_samples_kwargs: Check the parameters from `show_samples`
    """
    batch, records = batch_and_records

    for tensor_image, gt_masks, record in zip(batch[0][:], batch[1], records):
        image = tensor_to_image(tensor_image)
        record.set_img(image)

        mask = MaskArray(gt_masks.cpu().numpy().squeeze())
        record.segmentation.set_mask_array(mask)

    return show_samples(records, ncols=ncols, figsize=figsize, **show_samples_kwargs)
