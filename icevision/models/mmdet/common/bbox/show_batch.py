__all__ = ["show_batch"]

from icevision.utils import *
from icevision.visualize import *


def show_batch(batch_and_records, ncols: int = 1, figsize=None, **show_samples_kwargs):
    """Show a single batch from a dataloader.
    # Arguments
        show_samples_kwargs: Check the parameters from `show_samples`
    """
    batch, records = batch_and_records

    # In inference, both "img" and "img_metas" are lists. Check out the `build_infer_batch()` definition
    # We need to convert that to a batch similar to train and valid batches
    if isinstance(batch["img"], list):
        batch = {
            "img": batch["img"][0],
            "img_metas": batch["img_metas"][0],
        }

    for tensor_image, record in zip(batch["img"][:], records):
        image = (
            tensor_image.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1].copy()
        )
        record.set_img(image)

    return show_samples(records, ncols=ncols, figsize=figsize, **show_samples_kwargs)
