__all__ = ["show_batch"]

from icevision.utils import *
from icevision.visualize import *


def show_batch(batch_and_records, ncols: int = 1, figsize=None, **show_samples_kwargs):
    """Show a single batch from a dataloader.
    # Arguments
        show_samples_kwargs: Check the parameters from `show_samples`
    """
    batch, records = batch_and_records
    tensor_images = [img for img in batch['img'][0][:]]

    for tensor_image, record in zip(tensor_images, records):
        image = tensor_to_image(tensor_image)
        record.set_img(image)

    return show_samples(records, ncols=ncols, figsize=figsize, **show_samples_kwargs)