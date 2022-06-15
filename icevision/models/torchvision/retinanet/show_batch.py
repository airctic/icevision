__all__ = ["show_batch"]

from icevision.utils import *
from icevision.visualize import *
from icevision.models.utils import *


def show_batch(batch, ncols: int = 1, figsize=None, **show_samples_kwargs):
    """Show a single batch from a dataloader.

    # Arguments
        show_samples_kwargs: Check the parameters from `show_samples`
    """
    (tensor_images, labels), records = unpack_batch(batch)

    for tensor_image, label, record in zip(tensor_images, labels, records):
        image = tensor_to_image(tensor_image)
        record.set_img(image)

    return show_samples(records, ncols=ncols, figsize=figsize, **show_samples_kwargs)
