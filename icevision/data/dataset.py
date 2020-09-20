__all__ = ["Dataset"]

from icevision.imports import *
from icevision.tfms import *
from icevision.data.prepare_record import *


class Dataset:
    """Container for a list of records and transforms.

    Steps each time an item is requested (normally via directly indexing the `Dataset`):
        * Grab a record from the internal list of records.
        * Prepare the record (open the image, open the mask, add metadata).
        * Apply transforms to the record.

    # Arguments
        records: A list of records.
        tfm: Transforms to be applied to each item.
        prepare_record: Function that prepares the record before the transforms are applied.
    """

    def __init__(
        self,
        records: List[dict],
        tfm: Transform = None,
        prepare_record=None,
    ):
        self.records = records
        self.tfm = tfm
        self.prepare_record = prepare_record or default_prepare_record

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        data = self.prepare_record(self.records[i])
        if self.tfm is not None:
            data = self.tfm(data)
        return data

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items>"

    @classmethod
    def from_images(
        cls, images: Sequence[np.array], tfm: Transform = None, prepare_record=None
    ):
        """Creates a `Dataset` from a list of images.

        # Arguments
            images: `Sequence` of images in memory (numpy arrays).
            tfm: Transforms to be applied to each item.
            prepare_record: Function that prepares the record before the transforms are applied.
        """
        records = []
        for image in images:
            record = {"img": image}
            record["height"], record["width"], _ = image.shape
            records.append(record)

        return cls(records=records, tfm=tfm, prepare_record=prepare_record)
